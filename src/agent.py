from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState
from a2a.utils import new_agent_text_message

from purple_agent.io_bundle import (
    extract_competition_bundle,
    load_core_files,
    read_description,
)
from purple_agent.modeling import infer_task, solve_competition
from purple_agent.submission import (
    build_fallback_submission,
    sanitize_submission,
    validate_submission,
)


class Agent:
    def __init__(self) -> None:
        self.logs: list[str] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reading competition bundle..."),
        )

        with tempfile.TemporaryDirectory(prefix="purple-agent-") as tmpdir:
            workdir = Path(tmpdir)
            data_dir = extract_competition_bundle(message, workdir)

            description = read_description(data_dir)
            train_df, test_df, sample_df, paths_info = load_core_files(data_dir)
            self.logs.append(
                json.dumps(
                    {
                        "paths": paths_info,
                        "train_shape": list(train_df.shape),
                        "test_shape": list(test_df.shape),
                        "sample_shape": list(sample_df.shape),
                    },
                    ensure_ascii=False,
                )
            )

            baseline_submission_df = self._build_fallback_submission(
                test_df=test_df,
                sample_df=sample_df,
            )
            self._validate_submission(baseline_submission_df, sample_df)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Submitting safe baseline submission..."),
            )
            await self._add_submission_artifact(
                updater=updater,
                submission_df=baseline_submission_df,
                artifact_id="submission",
            )

            task = infer_task(train_df, test_df, sample_df)
            self.logs.append(json.dumps(task, ensure_ascii=False))

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Building features and evaluating candidate models..."),
            )

            try:
                submission_df, summary = self._solve_competition(
                    train_df=train_df,
                    test_df=test_df,
                    sample_df=sample_df,
                    description=description,
                    task=task,
                )
                self.logs.append(summary)
                self._validate_submission(submission_df, sample_df)

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

    def _solve_competition(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        description: str,
        task: dict[str, object],
    ) -> tuple[pd.DataFrame, str]:
        _ = description
        return solve_competition(
            train_df=train_df,
            test_df=test_df,
            sample_df=sample_df,
            task=task,
            logs=self.logs,
        )

    def _sanitize_submission(self, submission: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        return sanitize_submission(submission, sample_df)

    def _build_fallback_submission(self, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        return build_fallback_submission(test_df, sample_df)

    def _validate_submission(self, submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
        validate_submission(submission_df, sample_df)

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