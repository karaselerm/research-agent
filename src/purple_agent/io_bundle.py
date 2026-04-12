from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

import pandas as pd
from a2a.types import FilePart, FileWithBytes, Message


def extract_competition_bundle(message: Message, workdir: Path) -> Path:
    bundle_bytes = None

    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            file_obj = root.file
            if isinstance(file_obj, FileWithBytes) and file_obj.bytes is not None:
                raw = file_obj.bytes
                if isinstance(raw, str):
                    try:
                        bundle_bytes = base64.b64decode(raw)
                        break
                    except Exception:
                        continue
                elif isinstance(raw, (bytes, bytearray)):
                    bundle_bytes = bytes(raw)
                    break

    if bundle_bytes is None:
        raise ValueError("No competition tar.gz found in message.")

    extract_root = workdir / "bundle"
    extract_root.mkdir(parents=True, exist_ok=True)

    mode = "r:gz" if bundle_bytes[:2] == b"\x1f\x8b" else "r:"
    with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode=mode) as tar:
        tar.extractall(extract_root, filter="data")

    candidates = [
        extract_root / "home" / "data",
        extract_root / "data",
        extract_root,
    ]
    for path in candidates:
        if path.exists():
            csvs = list(path.rglob("*.csv"))
            if csvs:
                return path

    return extract_root


def read_description(data_dir: Path) -> str:
    for name in ["description.md", "description.txt", "README.md"]:
        p = data_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")[:20000]
    return ""


def _read_csv_any(path: Path) -> pd.DataFrame | None:
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def load_core_files(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    csv_paths = list(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise ValueError("No CSV files found in extracted bundle.")

    train_path = None
    test_path = None
    sample_path = None

    for p in csv_paths:
        s = str(p).lower()
        if "sample_submission" in s:
            sample_path = p
        elif s.endswith("train.csv") or "/train.csv" in s or "\\train.csv" in s:
            train_path = p
        elif s.endswith("test.csv") or "/test.csv" in s or "\\test.csv" in s:
            test_path = p

    if train_path is None:
        for p in csv_paths:
            s = str(p).lower()
            if "train" in s and "sample" not in s:
                train_path = p
                break

    if test_path is None:
        for p in csv_paths:
            s = str(p).lower()
            if "test" in s and "sample" not in s:
                test_path = p
                break

    if sample_path is None:
        for p in csv_paths:
            df = _read_csv_any(p)
            if df is None or df.shape[1] != 2:
                continue
            first, second = df.columns[0], df.columns[1]
            if first.lower().endswith("id") or second.lower() in {
                "target",
                "transported",
                "survived",
                "label",
            }:
                sample_path = p
                break

    if train_path is None or test_path is None or sample_path is None:
        raise ValueError(
            f"Could not identify train/test/sample files. "
            f"train={train_path}, test={test_path}, sample={sample_path}"
        )

    train_df = _read_csv_any(train_path)
    test_df = _read_csv_any(test_path)
    sample_df = _read_csv_any(sample_path)

    if train_df is None or test_df is None or sample_df is None:
        raise ValueError("Failed to read one or more CSV files.")

    return (
        train_df,
        test_df,
        sample_df,
        {
            "train": str(train_path),
            "test": str(test_path),
            "sample_submission": str(sample_path),
        },
    )