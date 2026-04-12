from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import queue
import re
import signal
import sys
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from pandas.api import types as pdt

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "14"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))

MAX_OUTPUT_CHARS = 4000
MAX_READ_FILE_CHARS = 50000
READ_FILE_SIZE_LIMIT_BYTES = 200_000
READ_FILE_CSV_SIZE_LIMIT_BYTES = 50_000
LARGE_TABULAR_ROWS = 200_000
LARGE_TABULAR_CELLS = 5_000_000
EVAL_SUBSAMPLE_ROWS = 120_000

SYSTEM_PROMPT = """\
You are an expert ML engineer solving a generic MLE-bench competition.

You can use these tools:
- `run_python(code: string)`: executes Python code in a persistent interpreter session.
- `list_files(path: string)`: lists files and directories relative to the workspace.
- `read_file(path: string, max_chars?: integer)`: reads a UTF-8 text file.
- `inspect_csv(path: string, max_rows?: integer)`: returns a compact JSON summary of a CSV file.
- `infer_tabular_task(spec_json: string)`: infers target candidates, task type, feature makeup, and validation strategy.
- `evaluate_tabular_candidates(spec_json: string)`: evaluates tabular model candidates on a single validation strategy.

Environment:
- Working directory contains the extracted competition bundle
- Competition files are typically under `./home/data/`
- Read the available files before making assumptions
- Save the final answer as `./submission.csv`

Required workflow:
1. List files under `./home/data/` and identify relevant inputs.
2. Read competition instructions from files such as `description.md` if present.
3. Inspect the sample submission and preserve its exact columns and ordering.
4. Use `infer_tabular_task` before training to infer task type, target candidates, and validation strategy.
5. Use `evaluate_tabular_candidates` to compare multiple validated candidates before choosing a final approach.
6. After choosing a simple validated approach, use `run_python` to materialize the final `submission.csv`.
7. Finish with a plain text response once the file is written.

Rules:
- Do not use competition-specific heuristics or hardcoded assumptions about column names, targets, metrics, or feature engineering before inspecting the data.
- Keep stdout concise. Print only the facts needed to debug progress.
- Prefer `list_files`, `read_file`, and `inspect_csv` for exploration.
- Never use `read_file` on large CSV files such as `train.csv`, `test.csv`, or `sample_submission.csv`; use `inspect_csv` for CSV inspection.
- Always inspect `sample_submission.csv` with `inspect_csv`, never `read_file`.
- For large tabular datasets, do not run heavy full-dataset experiments before selecting 1-2 candidates.
- For large tabular datasets, start with a subsample to choose the modeling family, then do one final fit on the full training data.
- For large tabular datasets with high-cardinality categoricals, avoid full one-hot encoding; prefer frequency/ordinal style encodings or tree-friendly pipelines.
- Avoid nested CV and avoid ensembles by default.
- Prefer the structured ML/eval tools for task inference and candidate comparison.
- If a structured tool returns an error for a candidate schema, do not repeat the same schema with minor variations; switch to a supported schema or use a simple `run_python` baseline.
- If a tool returns an error, treat it as feedback and recover in the next call instead of repeating the same mistake.
- Use `run_python` as the fallback and for final submission materialization, not as the first choice for model selection.
- Prefer reliable, generic code over fragile complexity.
- If an error occurs, inspect the traceback and fix it in the next tool call.
- Ensure the final `submission.csv` row count matches the test or sample submission.
- Use paths starting with `./`.
"""

_WARNING_BLOCK_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)


def _strip_warnings(text: str) -> str:
    return _WARNING_BLOCK_RE.sub("", text).strip()


def _truncate_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 80
    truncated_chars = len(text) - keep
    return f"[...{truncated_chars} chars truncated...]\n{text[-keep:]}"


def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _first_tar_from_message(message: Message) -> str | None:
    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            fd = root.file
            if isinstance(fd, FileWithBytes) and fd.bytes is not None:
                raw = fd.bytes
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, (bytes, bytearray)):
                    return base64.b64encode(raw).decode("ascii")
    return None


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _find_data_dir(workdir: Path) -> Path:
    inner = workdir / "home" / "data"
    return inner if inner.is_dir() else workdir


def _submission_debug_summary(submission_path: Path | None) -> str:
    if submission_path is None:
        return "submission_path=<none>"
    exists = submission_path.exists()
    if not exists:
        return f"submission_path={submission_path} exists=false"

    size_bytes = submission_path.stat().st_size
    preview = ""
    try:
        preview = submission_path.read_text(encoding="utf-8")[:400]
    except Exception as exc:
        preview = f"<failed to read preview: {exc}>"
    return (
        f"submission_path={submission_path} exists=true size_bytes={size_bytes}\n"
        f"preview:\n{preview}"
    )


def _infer_prediction_columns(sample: pd.DataFrame, test: pd.DataFrame | None) -> list[str]:
    if test is not None:
        predicted = [col for col in sample.columns if col not in test.columns]
        if predicted:
            return predicted
    if len(sample.columns) > 1:
        return sample.columns[1:].tolist()
    return sample.columns.tolist()


def _validate_column_family(column: str, sample_series: pd.Series, submission_series: pd.Series) -> None:
    if pdt.is_bool_dtype(sample_series):
        invalid = submission_series.dropna().map(lambda value: str(value).lower() not in {"true", "false", "0", "1"})
        if invalid.any():
            raise ValueError(f"Prediction column {column!r} must be boolean-like")
        return

    if pdt.is_integer_dtype(sample_series):
        if not pdt.is_numeric_dtype(submission_series):
            raise ValueError(f"Prediction column {column!r} must be numeric/integer-like")
        numeric = pd.to_numeric(submission_series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Prediction column {column!r} contains non-numeric values")
        fractional = (numeric % 1 != 0).fillna(False)
        if fractional.any():
            raise ValueError(f"Prediction column {column!r} must contain integer values")
        return

    if pdt.is_float_dtype(sample_series):
        if not pdt.is_numeric_dtype(submission_series):
            raise ValueError(f"Prediction column {column!r} must be numeric")
        numeric = pd.to_numeric(submission_series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Prediction column {column!r} contains non-numeric values")
        return


def _validate_submission_quality(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame | None,
) -> None:
    if submission.empty:
        raise ValueError("submission is empty")

    if submission.isna().any().any():
        nan_columns = submission.columns[submission.isna().any()].tolist()
        raise ValueError(f"submission contains NaN values in columns: {nan_columns}")

    id_candidates = [sample.columns[0]]
    if test is not None:
        shared = [col for col in sample.columns if col in test.columns]
        if shared:
            id_candidates = shared[:1]

    for id_col in id_candidates:
        if id_col in submission.columns and submission[id_col].duplicated().any():
            raise ValueError(f"submission contains duplicate values in id column {id_col!r}")

    prediction_columns = _infer_prediction_columns(sample, test)
    for column in prediction_columns:
        if column not in submission.columns:
            raise ValueError(f"prediction column {column!r} missing after normalization")
        _validate_column_family(column, sample[column], submission[column])


def _patch_submission_columns(workdir: Path, submission_path: Path) -> Path:
    data_dir = _find_data_dir(workdir)
    sample_path = _find_first(data_dir, "sample_submission*.csv")
    test_path = _find_first(data_dir, "test*.csv")

    if not submission_path.exists() or sample_path is None:
        return submission_path

    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path) if test_path is not None else None

    if len(submission) != len(sample):
        return submission_path

    missing_columns = [col for col in sample.columns if col not in submission.columns]
    extra_columns = [col for col in submission.columns if col not in sample.columns]

    expected_prediction_columns = _infer_prediction_columns(sample, test)
    missing_prediction_columns = [col for col in expected_prediction_columns if col not in submission.columns]
    extra_prediction_columns = [
        col for col in extra_columns if test is None or col not in test.columns
    ]

    if len(missing_prediction_columns) == 1 and len(extra_prediction_columns) == 1:
        submission = submission.rename(
            columns={extra_prediction_columns[0]: missing_prediction_columns[0]}
        )
        missing_columns = [col for col in sample.columns if col not in submission.columns]

    for col in missing_columns:
        if test is not None and col in test.columns:
            submission[col] = test[col].values
        elif col in sample.columns:
            submission[col] = sample[col].values

    if all(col in submission.columns for col in sample.columns):
        submission = submission.loc[:, sample.columns.tolist()]
        submission.to_csv(submission_path, index=False)

    return submission_path


def normalize_submission(workdir: Path, submission_path: Path) -> Path:
    data_dir = _find_data_dir(workdir)
    sample_path = _find_first(data_dir, "sample_submission*.csv")
    test_path = _find_first(data_dir, "test*.csv")

    if not submission_path.exists():
        raise FileNotFoundError(f"submission.csv not found at {submission_path}")

    if sample_path is None:
        logger.info("No sample submission found; skipping normalization")
        return submission_path

    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path) if test_path is not None else None

    expected_rows = len(test) if test is not None else len(sample)
    if len(submission) != expected_rows:
        raise ValueError(
            f"submission row count {len(submission)} does not match expected {expected_rows}"
        )

    missing_cols = [col for col in sample.columns if col not in submission.columns]
    for col in missing_cols:
        if test is not None and col in test.columns:
            submission[col] = test[col].values
        elif col in sample.columns:
            submission[col] = sample[col].values
        else:
            raise ValueError(f"Cannot restore missing submission column: {col}")

    still_missing = [col for col in sample.columns if col not in submission.columns]
    if still_missing:
        raise ValueError(f"Submission is still missing columns: {still_missing}")

    submission = submission[sample.columns.tolist()]
    _validate_submission_quality(submission, sample, test)
    submission.to_csv(submission_path, index=False)
    return submission_path


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


class OpenAIResponsesClient:
    def __init__(self, api_key: str, model: str = "gpt-5.4"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def create_initial_response(
        self,
        *,
        system_prompt: str,
        user_input: str,
        tools: list[dict[str, Any]],
    ):
        return self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input,
            tools=tools,
        )

    def create_followup_response(
        self,
        *,
        previous_response_id: str,
        tool_outputs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ):
        return self.client.responses.create(
            model=self.model,
            previous_response_id=previous_response_id,
            input=tool_outputs,
            tools=tools,
        )

    @staticmethod
    def extract_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def extract_tool_calls(response: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_args = getattr(item, "arguments", "{}") or "{}"
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_calls.append(
                ToolCall(
                    call_id=getattr(item, "call_id"),
                    name=getattr(item, "name"),
                    arguments=parsed_args,
                )
            )
        return tool_calls


@dataclass
class ExecutionResult:
    term_out: list[str]
    exec_time: float
    exc_type: str | None = None

    @property
    def output(self) -> str:
        return "".join(self.term_out)

    @property
    def timed_out(self) -> bool:
        return self.exc_type == "TimeoutError"


class _RedirectQueue:
    def __init__(self, q: Queue, timeout: float = 5.0):
        self.q = q
        self.timeout = timeout

    def write(self, msg: str) -> None:
        try:
            self.q.put(msg, timeout=self.timeout)
        except queue.Full:
            pass

    def flush(self) -> None:
        pass


def _run_session(working_dir: str, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
    os.chdir(working_dir)
    sys.path.insert(0, working_dir)
    sys.stdout = sys.stderr = _RedirectQueue(result_outq)

    global_scope: dict[str, Any] = {}
    while True:
        code = code_inq.get()
        os.chdir(working_dir)
        event_outq.put(("ready",))
        try:
            exec(compile(code, "<agent_code>", "exec"), global_scope)
        except BaseException as exc:
            lines = traceback.format_exception(exc)
            tb_str = "".join(
                line for line in lines if "agent.py" not in line and "importlib" not in line
            )
            exc_cls = type(exc).__name__
            if exc_cls == "KeyboardInterrupt":
                exc_cls = "TimeoutError"
            result_outq.put(tb_str)
            event_outq.put(("finished", exc_cls))
        else:
            event_outq.put(("finished", None))

        result_outq.put("<|EOF|>")


class Interpreter:
    def __init__(self, workdir: str | Path, timeout: int = 600):
        self.working_dir = str(Path(workdir).resolve())
        self.timeout = timeout
        self._process: Process | None = None
        self._code_inq: Queue | None = None
        self._result_outq: Queue | None = None
        self._event_outq: Queue | None = None

    def create_process(self) -> None:
        self._code_inq = Queue()
        self._result_outq = Queue()
        self._event_outq = Queue()
        self._process = Process(
            target=_run_session,
            args=(self.working_dir, self._code_inq, self._result_outq, self._event_outq),
            daemon=True,
        )
        self._process.start()

    def cleanup(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.join(timeout=2.0)
            if self._process.exitcode is None:
                self._process.kill()
                self._process.join(timeout=1.0)
        except Exception as exc:
            logger.error("Error cleaning up interpreter process: %s", exc)
        finally:
            self._process = None

    def run(self, code: str, reset_session: bool = False) -> ExecutionResult:
        if reset_session or self._process is None:
            self.cleanup()
            self.create_process()

        assert self._process is not None and self._process.is_alive()
        assert self._code_inq is not None
        assert self._event_outq is not None
        assert self._result_outq is not None

        self._code_inq.put(code)

        try:
            event = self._event_outq.get(timeout=10)
        except queue.Empty as exc:
            raise RuntimeError("Interpreter child process failed to start execution") from exc
        assert event[0] == "ready", event

        start = time.time()
        overtime = False

        while True:
            try:
                event = self._event_outq.get(timeout=1.0)
                assert event[0] == "finished", event
                exec_time = time.time() - start
                exc_type = event[1]
                break
            except queue.Empty:
                if not overtime and not self._process.is_alive():
                    raise RuntimeError("Interpreter process died unexpectedly")

                elapsed = time.time() - start
                if elapsed > self.timeout:
                    logger.warning("Execution exceeded timeout of %ds", self.timeout)
                    os.kill(self._process.pid, signal.SIGINT)
                    overtime = True

                if overtime and (time.time() - start) > self.timeout + 10:
                    self.cleanup()
                    exec_time = self.timeout
                    exc_type = "TimeoutError"
                    break

        output: list[str] = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                chunk = self._result_outq.get(timeout=0.5)
                if chunk == "<|EOF|>":
                    break
                output.append(chunk)
            except queue.Empty:
                continue

        if exc_type == "TimeoutError":
            output.append(f"\nTimeoutError: execution exceeded {self.timeout}s time limit\n")
        else:
            output.append(f"\n[Execution time: {exec_time:.1f}s]\n")

        return ExecutionResult(term_out=output, exec_time=exec_time, exc_type=exc_type)

    def __del__(self) -> None:
        self.cleanup()


class MLAgent:
    def __init__(
        self,
        workdir: str | Path,
        api_key: str,
        model: str = "gpt-5.4",
        max_iterations: int = 24,
        code_timeout: int = 600,
        updater: TaskUpdater | None = None,
        llm_client: OpenAIResponsesClient | None = None,
    ):
        self.workdir = Path(workdir).resolve()
        self.max_iterations = max_iterations
        self.updater = updater
        self._loop: asyncio.AbstractEventLoop | None = None
        self._python_session_started = False
        self._artifact_counter = 0
        self._artifacts: dict[str, Any] = {}
        self.interpreter = Interpreter(workdir=self.workdir, timeout=code_timeout)
        self.llm = llm_client or OpenAIResponsesClient(api_key=api_key, model=model)

    def _post_status(self, text: str) -> None:
        if self.updater is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.updater.update_status(TaskState.working, new_agent_text_message(text)),
            self._loop,
        )

    def _resolve_path(self, path: str) -> Path:
        raw_path = Path(path or ".")
        candidate = (self.workdir / raw_path).resolve()
        if candidate != self.workdir and self.workdir not in candidate.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return candidate
    def _parse_spec_json(self, spec_json: str) -> dict[str, Any]:
        spec = json.loads(spec_json)
        if not isinstance(spec, dict):
            raise ValueError("spec_json must decode to an object")
        return spec

    def _ok_response(self, **payload: Any) -> str:
        body = {"status": "ok", "error": None, **payload}
        return json.dumps(body, ensure_ascii=True, separators=(",", ":"))

    def _error_response(self, error: str, **payload: Any) -> str:
        body = {"status": "error", "error": error, **payload}
        return json.dumps(body, ensure_ascii=True, separators=(",", ":"))

    def _store_artifact(self, payload: Any, prefix: str) -> str:
        self._artifact_counter += 1
        ref = f"{prefix}_{self._artifact_counter}"
        self._artifacts[ref] = payload
        return ref

    def _tool_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "run_python",
                "description": "Execute Python code in a persistent interpreter session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_files",
                "description": "List files and directories under a workspace-relative path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a UTF-8 text file under the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "inspect_csv",
                "description": "Return a compact JSON summary for a CSV file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_rows": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "infer_tabular_task",
                "description": "Infer target candidates, task type, feature makeup, and recommended validation strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {"spec_json": {"type": "string"}},
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "evaluate_tabular_candidates",
                "description": "Evaluate tabular model candidates on a shared validation strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {"spec_json": {"type": "string"}},
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
        ]

    def _run_python(self, code: str, *, reset_session: bool = False) -> str:
        result = self.interpreter.run(code, reset_session=reset_session)
        clean_output = _strip_warnings(result.output)
        return _truncate_output(clean_output, MAX_OUTPUT_CHARS)

    def _list_files(self, path: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if target.is_file():
            rel = target.relative_to(self.workdir)
            return str(rel)

        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            rel = child.relative_to(self.workdir)
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{rel}{suffix}")
        return "\n".join(entries) if entries else "<empty directory>"

    def _read_file(self, path: str, max_chars: int = 12000) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {path}")

        suffix = target.suffix.lower()
        size_bytes = target.stat().st_size
        if suffix == ".csv" and size_bytes > READ_FILE_CSV_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"Refusing to read large CSV file via read_file: {path} ({size_bytes} bytes). Use inspect_csv instead."
            )
        if size_bytes > READ_FILE_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"Refusing to read large file via read_file: {path} ({size_bytes} bytes). Use inspect_csv for CSVs or request a smaller text file."
            )

        text = target.read_text(encoding="utf-8")
        return _truncate_output(text, max(200, min(max_chars, MAX_READ_FILE_CHARS)))

    def _inspect_csv(self, path: str, max_rows: int = 5) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {path}")

        frame = pd.read_csv(target)
        preview_rows = frame.head(max(1, min(max_rows, 20))).to_dict(orient="records")
        summary = {
            "path": str(target.relative_to(self.workdir)),
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "preview": preview_rows,
        }
        return _truncate_output(json.dumps(summary, ensure_ascii=True, indent=2), MAX_OUTPUT_CHARS)

    def _load_tabular_bundle(self, spec: dict[str, Any]):
        train = pd.read_csv(self._resolve_path(str(spec["train_path"])))
        test_path = spec.get("test_path")
        sample_path = spec.get("sample_submission_path")
        test = pd.read_csv(self._resolve_path(str(test_path))) if test_path else None
        sample = pd.read_csv(self._resolve_path(str(sample_path))) if sample_path else None
        return train, test, sample

    def _infer_target_candidates(self, train, test, sample) -> list[str]:
        if sample is not None and test is not None:
            sample_only = [col for col in sample.columns if col not in test.columns]
            if sample_only:
                return sample_only
        if test is not None:
            train_only = [col for col in train.columns if col not in test.columns]
            if train_only:
                return train_only
        return train.columns[-1:].tolist()

    def _infer_id_candidates(self, train, test, sample) -> list[str]:
        candidates: list[str] = []
        if test is not None and sample is not None:
            candidates.extend([col for col in sample.columns if col in test.columns])
        if test is not None:
            for col in test.columns:
                lower = str(col).lower()
                if lower.endswith("id") or lower == "id":
                    candidates.append(col)
        seen = set()
        ordered = []
        for item in candidates:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered[:5]

    def _detect_text_datetime_features(self, frame, exclude: set[str]) -> tuple[dict[str, Any], bool, bool]:
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []

        for column in frame.columns:
            if column in exclude:
                continue
            series = frame[column]
            if str(series.dtype).startswith(("int", "float", "bool")):
                numeric_cols.append(column)
                continue
            categorical_cols.append(column)
            sample = series.dropna().astype(str).head(100)
            avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
            if avg_len >= 20.0:
                text_cols.append(column)
            parsed = None
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
            except Exception:
                parsed = None
            if parsed is not None and len(sample) > 0 and float(parsed.notna().mean()) >= 0.8:
                datetime_cols.append(column)

        feature_summary = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "text_columns": text_cols,
            "datetime_columns": datetime_cols,
            "column_count": int(len(frame.columns) - len(exclude)),
        }
        return feature_summary, bool(text_cols), bool(datetime_cols)

    def _infer_task_type_from_target(self, target_series) -> str:
        if target_series.dtype == object or str(target_series.dtype).startswith("string"):
            nunique = int(target_series.nunique(dropna=True))
            return "binary_classification" if nunique <= 2 else "multiclass_classification"
        if pd.api.types.is_bool_dtype(target_series):
            return "binary_classification"
        if pd.api.types.is_integer_dtype(target_series):
            nunique = int(target_series.nunique(dropna=True))
            if nunique <= 20:
                return "binary_classification" if nunique <= 2 else "multiclass_classification"
        return "regression"

    def _recommended_metric_family(self, task_type: str) -> str:
        if task_type in {"binary_classification", "multiclass_classification"}:
            return "accuracy"
        return "neg_rmse"

    def _recommended_validation(self, task_type: str, datetime_heavy: bool, group_candidates: list[str]) -> str:
        if datetime_heavy:
            return "time_split"
        if group_candidates:
            return "grouped_kfold_recommended"
        if "classification" in task_type:
            return "stratified_kfold"
        return "kfold"

    def _infer_tabular_task(self, spec_json: str) -> str:
        try:
            spec = self._parse_spec_json(spec_json)
            train, test, sample = self._load_tabular_bundle(spec)
            target_candidates = self._infer_target_candidates(train, test, sample)
            id_candidates = self._infer_id_candidates(train, test, sample)
            chosen_target = str(spec.get("target_column") or target_candidates[0])
            exclude = set(id_candidates + [chosen_target])
            feature_summary, text_heavy, datetime_heavy = self._detect_text_datetime_features(train, exclude)
            task_type = str(spec.get("task_type") or self._infer_task_type_from_target(train[chosen_target]))
            validation = self._recommended_validation(task_type, datetime_heavy, id_candidates)
            return self._ok_response(
                task_type=task_type,
                target_candidates=target_candidates,
                id_candidates=id_candidates,
                feature_summary=feature_summary,
                recommended_validation=validation,
                recommended_metric_family=self._recommended_metric_family(task_type),
                text_heavy=text_heavy,
                datetime_heavy=datetime_heavy,
                available_candidates={
                    "binary_classification": [
                        "logistic_regression",
                        "lightgbm_classifier",
                        "catboost_classifier",
                        "xgboost_classifier",
                    ],
                    "multiclass_classification": [
                        "logistic_regression",
                        "lightgbm_classifier",
                        "catboost_classifier",
                        "xgboost_classifier",
                    ],
                    "regression": [
                        "linear_regression",
                        "lightgbm_regressor",
                        "catboost_regressor",
                        "xgboost_regressor",
                    ],
                    "text_heavy_tabular": ["tfidf_linear", "logistic_regression", "lightgbm_classifier"],
                },
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _default_candidates(self, task_type: str, text_heavy: bool) -> list[str]:
        if task_type == "regression":
            return ["linear_regression", "lightgbm_regressor"]
        if text_heavy:
            return ["tfidf_linear", "logistic_regression", "lightgbm_classifier"]
        return ["logistic_regression", "lightgbm_classifier"]

    def _is_large_tabular_dataset(self, frame) -> bool:
        rows = int(len(frame))
        cols = int(len(frame.columns))
        return rows >= LARGE_TABULAR_ROWS or rows * max(cols, 1) >= LARGE_TABULAR_CELLS

    def _subsample_training_frame(self, train, target_column: str, task_type: str):
        if not self._is_large_tabular_dataset(train) or len(train) <= EVAL_SUBSAMPLE_ROWS:
            return train, False

        if "classification" in task_type and target_column in train.columns:
            grouped = []
            target = train[target_column]
            for _, group in train.groupby(target, dropna=False):
                take = max(1, int(round(len(group) / len(train) * EVAL_SUBSAMPLE_ROWS)))
                grouped.append(group.sample(n=min(len(group), take), random_state=42))
            sampled = pd.concat(grouped, axis=0).sample(frac=1.0, random_state=42)
            if len(sampled) > EVAL_SUBSAMPLE_ROWS:
                sampled = sampled.head(EVAL_SUBSAMPLE_ROWS)
            return sampled.reset_index(drop=True), True

        sampled = train.sample(n=EVAL_SUBSAMPLE_ROWS, random_state=42)
        return sampled.reset_index(drop=True), True

    def _apply_feature_plan(self, train_df, test_df, feature_plan: dict[str, Any], target_column: str):
        import numpy as np

        train = train_df.copy()
        test = test_df.copy() if test_df is not None else None
        transforms = feature_plan.get("transforms_applied", [])

        for col in feature_plan.get("missing_indicators", []):
            if col in train.columns:
                train[f"{col}__is_missing"] = train[col].isna().astype(int)
                if test is not None and col in test.columns:
                    test[f"{col}__is_missing"] = test[col].isna().astype(int)

        for col in feature_plan.get("datetime_parts", []):
            if col in train.columns:
                tr = pd.to_datetime(train[col], errors="coerce")
                train[f"{col}__year"] = tr.dt.year
                train[f"{col}__month"] = tr.dt.month
                train[f"{col}__day"] = tr.dt.day
                if test is not None and col in test.columns:
                    te = pd.to_datetime(test[col], errors="coerce")
                    test[f"{col}__year"] = te.dt.year
                    test[f"{col}__month"] = te.dt.month
                    test[f"{col}__day"] = te.dt.day

        for col in feature_plan.get("log1p_columns", []):
            if col in train.columns:
                train[col] = pd.to_numeric(train[col], errors="coerce")
                train[f"{col}__log1p"] = np.log1p(train[col].clip(lower=0))
                if test is not None and col in test.columns:
                    test[col] = pd.to_numeric(test[col], errors="coerce")
                    test[f"{col}__log1p"] = np.log1p(test[col].clip(lower=0))

        for col in feature_plan.get("frequency_encode", []):
            if col in train.columns:
                freq = train[col].astype(str).value_counts(dropna=False).to_dict()
                train[f"{col}__freq"] = train[col].astype(str).map(freq).fillna(0)
                if test is not None and col in test.columns:
                    test[f"{col}__freq"] = test[col].astype(str).map(freq).fillna(0)

        if target_column in train.columns:
            train[target_column] = train_df[target_column]

        return train, test, transforms

    def _build_candidate_model(self, candidate_name: str, task_type: str, X, text_column: str | None):
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]
        large_dataset = self._is_large_tabular_dataset(X)
        low_memory_categoricals = large_dataset or len(categorical_cols) >= 8

        def _tabular_preprocessor(scale_numeric: bool) -> ColumnTransformer:
            numeric_steps = [("impute", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scale", StandardScaler()))
            transformers = []
            if numeric_cols:
                transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
            if categorical_cols:
                if low_memory_categoricals:
                    cat_steps = [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                else:
                    cat_steps = [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10 if large_dataset else None,
                            ),
                        ),
                    ]
                transformers.append(("cat", Pipeline(cat_steps), categorical_cols))
            return ColumnTransformer(transformers, remainder="drop")

        if candidate_name == "logistic_regression":
            estimator = LogisticRegression(max_iter=250, solver="lbfgs")
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=True)), ("model", estimator)])
        if candidate_name == "linear_regression":
            estimator = __import__("sklearn.linear_model").linear_model.LinearRegression()
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=True)), ("model", estimator)])
        if candidate_name == "lightgbm_classifier":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                n_estimators=80 if large_dataset else 120,
                learning_rate=0.07,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "lightgbm_regressor":
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(
                n_estimators=80 if large_dataset else 120,
                learning_rate=0.07,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "xgboost_classifier":
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=80 if large_dataset else 120,
                learning_rate=0.07,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                tree_method="hist",
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "xgboost_regressor":
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                n_estimators=80 if large_dataset else 120,
                learning_rate=0.07,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                tree_method="hist",
                random_state=42,
                verbosity=0,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "catboost_classifier":
            from catboost import CatBoostClassifier

            estimator = CatBoostClassifier(
                iterations=80 if large_dataset else 120,
                learning_rate=0.07,
                depth=5,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "catboost_regressor":
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(
                iterations=80 if large_dataset else 120,
                learning_rate=0.07,
                depth=5,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "tfidf_linear":
            if text_column is None:
                raise ValueError("tfidf_linear requires a text column")
            if task_type == "regression":
                from sklearn.linear_model import LinearRegression

                estimator = LinearRegression()
            else:
                estimator = LogisticRegression(max_iter=500)
            return Pipeline(
                [
                    (
                        "features",
                        __import__("sklearn.compose").compose.ColumnTransformer(
                            [
                                ("text", __import__("sklearn.feature_extraction.text").feature_extraction.text.TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_column),
                            ],
                            remainder="drop",
                        ),
                    ),
                    ("model", estimator),
                ]
            )
        raise ValueError(f"Unsupported candidate: {candidate_name}")

    def _normalize_candidate_name(self, candidate: Any) -> str:
        if isinstance(candidate, str):
            raw_name = candidate
        elif isinstance(candidate, dict):
            raw_name = ""
            for key in ("candidate_name", "model", "model_type", "name"):
                value = candidate.get(key)
                if isinstance(value, str) and value:
                    raw_name = value
                    break
            if not raw_name:
                raise ValueError(f"Unsupported candidate: {candidate}")
        else:
            raise ValueError(f"Unsupported candidate: {candidate}")

        aliases = {
            "logreg": "logistic_regression",
            "logistic": "logistic_regression",
            "linear": "linear_regression",
            "lgbm": "lightgbm_classifier",
            "lightgbm": "lightgbm_classifier",
            "catboost": "catboost_classifier",
            "xgboost": "xgboost_classifier",
            "tfidf": "tfidf_linear",
        }
        return aliases.get(raw_name, raw_name)

    def _evaluate_tabular_candidates(self, spec_json: str) -> str:
        try:
            from sklearn.metrics import accuracy_score, mean_squared_error
            from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

            spec = self._parse_spec_json(spec_json)
            train, test, sample = self._load_tabular_bundle(spec)
            target_candidates = self._infer_target_candidates(train, test, sample)
            target_column = str(spec.get("target_column") or target_candidates[0])
            task_type = str(spec.get("task_type") or self._infer_task_type_from_target(train[target_column]))
            train_eval_source, used_subsample = self._subsample_training_frame(train, target_column, task_type)

            id_candidates = self._infer_id_candidates(train, test, sample)
            feature_summary, text_heavy, datetime_heavy = self._detect_text_datetime_features(
                train_eval_source, set(id_candidates + [target_column])
            )
            validation_scheme = str(spec.get("validation_scheme") or self._recommended_validation(task_type, datetime_heavy, id_candidates))
            metric_family = str(spec.get("metric_family") or self._recommended_metric_family(task_type))
            feature_plan = spec.get("feature_plan") or {}

            train_eval, test_eval, transforms = self._apply_feature_plan(train_eval_source, test, feature_plan, target_column)
            X = train_eval.drop(columns=[target_column], errors="ignore")
            X = X.drop(columns=[col for col in id_candidates if col in X.columns], errors="ignore")
            X_test = None
            if test_eval is not None:
                X_test = test_eval.drop(columns=[col for col in id_candidates if col in test_eval.columns], errors="ignore")

            y = train_eval[target_column]
            text_column = feature_summary["text_columns"][0] if feature_summary["text_columns"] else None
            raw_candidates = spec.get("candidates") or self._default_candidates(task_type, text_heavy)
            candidate_names = [self._normalize_candidate_name(candidate) for candidate in raw_candidates]
            n_splits = int(spec.get("n_splits", 3))

            if validation_scheme == "time_split":
                splitter = TimeSeriesSplit(n_splits=max(2, n_splits))
                split_iter = splitter.split(X)
            elif "classification" in task_type:
                split_iter = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42).split(X, y)
            else:
                split_iter = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=42).split(X, y)

            split_indices = list(split_iter)
            candidates_out = []
            best_candidate = None
            best_score = None
            prediction_mode = "numeric" if task_type == "regression" else "label"

            for candidate_name in candidate_names:
                candidate_model = self._build_candidate_model(candidate_name, task_type, X, text_column)
                fold_scores: list[float] = []
                oof_predictions: list[Any] = [None] * len(X)
                oof_probabilities: list[Any] | None = [None] * len(X) if task_type == "binary_classification" else None

                for train_idx, valid_idx in split_indices:
                    X_tr = X.iloc[train_idx]
                    X_va = X.iloc[valid_idx]
                    y_tr = y.iloc[train_idx]
                    y_va = y.iloc[valid_idx]
                    candidate_model.fit(X_tr, y_tr)

                    if task_type == "regression":
                        pred = candidate_model.predict(X_va)
                        score = -float(mean_squared_error(y_va, pred) ** 0.5)
                        for local_idx, value in zip(valid_idx, pred):
                            oof_predictions[local_idx] = float(value)
                    elif task_type == "binary_classification":
                        if hasattr(candidate_model, "predict_proba"):
                            proba = candidate_model.predict_proba(X_va)[:, 1]
                            pred = (proba >= 0.5).astype(int)
                        else:
                            pred = candidate_model.predict(X_va)
                            proba = pred.astype(float)
                        score = float(accuracy_score(y_va, pred))
                        prediction_mode = "proba"
                        for local_idx, p_val, y_val in zip(valid_idx, proba, pred):
                            oof_probabilities[local_idx] = float(p_val)
                            oof_predictions[local_idx] = int(y_val)
                    else:
                        pred = candidate_model.predict(X_va)
                        score = float(accuracy_score(y_va, pred))
                        for local_idx, value in zip(valid_idx, pred):
                            oof_predictions[local_idx] = value.item() if hasattr(value, "item") else value

                    fold_scores.append(score)

                candidate_model.fit(X, y)
                test_output = None
                test_probability = None
                if X_test is not None:
                    if task_type == "regression":
                        pred = candidate_model.predict(X_test)
                        test_output = [float(item) for item in pred]
                    elif task_type == "binary_classification":
                        if hasattr(candidate_model, "predict_proba"):
                            test_probability = candidate_model.predict_proba(X_test)[:, 1]
                            test_output = [int(item) for item in (test_probability >= 0.5)]
                            test_probability = [float(item) for item in test_probability]
                        else:
                            pred = candidate_model.predict(X_test)
                            test_output = [int(item) for item in pred]
                    else:
                        pred = candidate_model.predict(X_test)
                        test_output = [item.item() if hasattr(item, "item") else item for item in pred]

                artifact_ref = self._store_artifact(
                    {
                        "task_type": task_type,
                        "metric_family": metric_family,
                        "candidate_name": candidate_name,
                        "validation_scheme": validation_scheme,
                        "y_true": [item.item() if hasattr(item, "item") else item for item in y.tolist()],
                        "oof_predictions": oof_predictions,
                        "oof_probabilities": oof_probabilities,
                        "test_predictions": test_output,
                        "test_probabilities": test_probability,
                        "feature_plan": feature_plan,
                        "target_column": target_column,
                        "id_candidates": id_candidates,
                        "used_subsample_for_eval": used_subsample,
                        "eval_rows": int(len(train_eval)),
                    },
                    "candidate",
                )

                candidate_result = {
                    "name": candidate_name,
                    "score": float(sum(fold_scores) / len(fold_scores)),
                    "fold_scores": [float(score) for score in fold_scores],
                    "artifact_ref": artifact_ref,
                    "fit_notes": [],
                }
                candidates_out.append(candidate_result)

                if best_score is None or candidate_result["score"] > best_score:
                    best_score = candidate_result["score"]
                    best_candidate = candidate_result

            candidates_out.sort(key=lambda item: item["score"], reverse=True)
            return self._ok_response(
                task_type=task_type,
                validation_scheme=validation_scheme,
                metric_family=metric_family,
                candidates=candidates_out,
                best_candidate=best_candidate,
                oof_available=True,
                prediction_mode=prediction_mode,
                feature_policy={"applied": transforms, "feature_plan": feature_plan},
                text_heavy=text_heavy,
                datetime_heavy=datetime_heavy,
                used_subsample_for_eval=used_subsample,
                eval_rows=int(len(train_eval)),
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _execute_tool(self, call, *, iteration: int, index: int) -> str:
        try:
            if call.name == "run_python":
                code = str(call.arguments.get("code", ""))
                preview = code.strip().splitlines()[0][:100] if code.strip() else "<empty>"
                self._post_status(f"run_python: {preview}")
                reset_session = not self._python_session_started
                output = self._run_python(code, reset_session=reset_session)
                self._python_session_started = True
                return output
            if call.name == "list_files":
                path = str(call.arguments.get("path", "."))
                self._post_status(f"list_files: {path}")
                return self._list_files(path)
            if call.name == "read_file":
                path = str(call.arguments.get("path", ""))
                max_chars = int(call.arguments.get("max_chars", 12000))
                self._post_status(f"read_file: {path}")
                return self._read_file(path, max_chars=max_chars)
            if call.name == "inspect_csv":
                path = str(call.arguments.get("path", ""))
                max_rows = int(call.arguments.get("max_rows", 5))
                self._post_status(f"inspect_csv: {path}")
                return self._inspect_csv(path, max_rows=max_rows)
            if call.name == "infer_tabular_task":
                self._post_status("infer_tabular_task")
                return self._infer_tabular_task(str(call.arguments.get("spec_json", "{}")))
            if call.name == "evaluate_tabular_candidates":
                self._post_status("evaluate_tabular_candidates")
                return self._evaluate_tabular_candidates(str(call.arguments.get("spec_json", "{}")))
            raise ValueError(f"Unsupported tool call: {call.name}")
        except Exception as exc:
            logger.warning("Tool call failed (%s): %s", call.name, exc)
            error_hint = ""
            if call.name == "read_file":
                error_hint = " Use inspect_csv for CSV files, especially sample_submission.csv, train.csv, and test.csv."
            return self._error_response(
                str(exc) + error_hint,
                tool_name=call.name,
                iteration=iteration,
                tool_index=index,
            )

    def _finalize_submission_if_possible(self) -> Path | None:
        submission = self.workdir / "submission.csv"
        finalizer_code = '''
from pathlib import Path
import json
import pandas as pd

workdir = Path(".")
submission_path = workdir / "submission.csv"
data_dir = workdir / "home" / "data"
if not data_dir.exists():
    data_dir = workdir

sample_candidates = sorted(data_dir.glob("sample_submission*.csv"))
test_candidates = sorted(data_dir.glob("test*.csv"))
sample_path = sample_candidates[0] if sample_candidates else None
test_path = test_candidates[0] if test_candidates else None

def _expected_rows(sample_df, test_df):
    return len(test_df) if test_df is not None else len(sample_df)

sample_df = pd.read_csv(sample_path) if sample_path is not None else None
test_df = pd.read_csv(test_path) if test_path is not None else None

if not submission_path.exists():
    namespace = globals()
    chosen_df = None

    if sample_df is not None:
        expected_rows = _expected_rows(sample_df, test_df)
        preferred_names = [
            "submission",
            "submission_df",
            "predictions_df",
            "pred_df",
            "result_df",
            "output_df",
            "final_submission",
        ]
        for name in preferred_names + sorted(namespace.keys()):
            obj = namespace.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) == expected_rows:
                chosen_df = obj.copy()
                break

        if chosen_df is None:
            for name in ["predictions", "preds", "y_pred", "test_predictions"]:
                obj = namespace.get(name)
                if obj is None:
                    continue
                try:
                    values = list(obj)
                except Exception:
                    continue
                if len(values) == expected_rows and len(sample_df.columns) >= 2:
                    chosen_df = sample_df.copy()
                    chosen_df[sample_df.columns[-1]] = values
                    break

    if chosen_df is not None:
        chosen_df.to_csv(submission_path, index=False)

summary = {"exists": submission_path.exists(), "path": str(submission_path)}
if submission_path.exists():
    preview_df = pd.read_csv(submission_path)
    summary["rows"] = int(len(preview_df))
    summary["columns"] = preview_df.columns.tolist()
print(json.dumps(summary, ensure_ascii=True))
'''
        try:
            output = self._run_python(finalizer_code, reset_session=not self._python_session_started)
            self._python_session_started = True
            logger.info("Deterministic finalizer output: %s", output)
        except Exception as exc:
            logger.warning("Deterministic finalizer failed: %s", exc)
        return submission if submission.exists() else None

    def run(self, instructions: str, loop: asyncio.AbstractEventLoop | None = None) -> Path | None:
        self._loop = loop
        self._python_session_started = False
        tools = self._tool_spec()

        try:
            self._post_status(f"Starting ML agent with model={self.llm.model}")
            response = self.llm.create_initial_response(
                system_prompt=SYSTEM_PROMPT,
                user_input=instructions or "Solve the competition and produce ./submission.csv",
                tools=tools,
            )

            for iteration in range(1, self.max_iterations + 1):
                tool_calls = self.llm.extract_tool_calls(response)
                if not tool_calls:
                    final_text = self.llm.extract_text(response)
                    if final_text:
                        logger.info("LLM finished: %s", final_text)
                    break

                self._post_status(f"Iteration {iteration}/{self.max_iterations}: executing tool calls")
                tool_outputs: list[dict[str, str]] = []
                for index, call in enumerate(tool_calls):
                    output = self._execute_tool(call, iteration=iteration, index=index)
                    logger.info("Tool output (%s): %s", call.name, output[:500])
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )

                response = self.llm.create_followup_response(
                    previous_response_id=response.id,
                    tool_outputs=tool_outputs,
                    tools=tools,
                )
            else:
                logger.warning("Max iterations reached without explicit finish")

            submission = self.workdir / "submission.csv"
            if not submission.exists():
                self._post_status("Finalizing submission.csv deterministically")
                self._finalize_submission_if_possible()
        finally:
            self.interpreter.cleanup()

        submission = self.workdir / "submission.csv"
        return submission if submission.exists() else None

class Agent:
    def __init__(self, *, ml_agent_cls=MLAgent):
        self._done_context: set[str] = set()
        self._ml_agent_cls = ml_agent_cls
        self.logs: list[str] = []

    def _build_fallback_submission(self, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        fallback = sample_df.copy()
        id_col = sample_df.columns[0]

        if id_col in test_df.columns:
            n = min(len(fallback), len(test_df))
            fallback.iloc[:n, 0] = test_df.iloc[:n][id_col].values

        return fallback
    def _validate_submission(self, submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
        if list(submission_df.columns) != list(sample_df.columns):
            raise ValueError(
                f"Expected columns {list(sample_df.columns)}, got {list(submission_df.columns)}"
            )
        if len(submission_df) != len(sample_df):
            raise ValueError(
                f"Expected {len(sample_df)} rows, got {len(submission_df)}"
            )
        if submission_df.isnull().any().any():
            raise ValueError("Submission contains NaN values")

    def _solve_competition(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        description: str,
        task: dict[str, object],
    ) -> tuple[pd.DataFrame, str]:
        _ = (train_df, test_df, sample_df, description, task)
        raise NotImplementedError("Compatibility wrapper is not implemented for the one-file agent")

    async def _add_submission_artifact(
        self,
        updater: TaskUpdater,
        submission_df: pd.DataFrame,
        artifact_id: str = "submission",
    ) -> None:
        csv_bytes = submission_df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv_bytes).decode("ascii")

        await updater.add_artifact(
            parts=[
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=b64,
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                )
            ],
            artifact_id=artifact_id,
            name="submission.csv",
            append=False,
            last_chunk=True,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        text = get_message_text(message)

        if ctx in self._done_context:
            logger.info("Context %s already finished; ack", ctx)
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            logger.error("No competition tar.gz in message; text len=%s", len(text))
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: expected FilePart competition.tar.gz"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Extracting competition bundle for context {ctx}..."),
        )

        with tempfile.TemporaryDirectory(prefix=f"mle-bench-{ctx}-") as temp_dir:
            work_dir = Path(temp_dir)

            try:
                _extract_tar_b64(tar_b64, work_dir)
            except Exception as exc:
                logger.exception("Extract failed")
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error extracting tar: {exc}"))],
                    name="Error",
                )
                return

            data_dir = _find_data_dir(work_dir)
            description_path = _find_first(data_dir, "description.md") or _find_first(data_dir, "README.md")
            description = ""
            if description_path and description_path.exists():
                description = description_path.read_text(encoding="utf-8", errors="replace")[:20000]

            train_path = _find_first(data_dir, "train.csv")
            test_path = _find_first(data_dir, "test.csv")
            sample_path = _find_first(data_dir, "sample_submission.csv")

            if train_path is None or test_path is None or sample_path is None:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: could not find train/test/sample_submission files"))],
                    name="Error",
                )
                return

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            sample_df = pd.read_csv(sample_path)

            baseline_submission_df = self._build_fallback_submission(
                test_df=test_df,
                sample_df=sample_df,
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Submitting safe baseline submission..."),
            )
            await self._add_submission_artifact(
                updater=updater,
                submission_df=baseline_submission_df,
                artifact_id="submission",
            )

            task = {"target_col": None, "id_col": sample_df.columns[0], "pred_col": sample_df.columns[1], "task_type": None}

            try:
                submission_df, _summary = self._solve_competition(
                    train_df=train_df,
                    test_df=test_df,
                    sample_df=sample_df,
                    description=description,
                    task=task,
                )

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("Uploading improved submission..."),
                )
                await self._add_submission_artifact(
                    updater=updater,
                    submission_df=submission_df,
                    artifact_id="submission",
                )
            except Exception as e:
                self.logs.append(f"fallback_used: {e}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "Model training failed or suspicious submission detected; using safe baseline submission."
                    ),
                )

        self._done_context.add(ctx)