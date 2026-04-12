from __future__ import annotations

import base64
import io
import json
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SEPARATORS = ["_", "/", "-", "|", ":"]


def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if raw[:2] == b"\x1f\x8b" else "r:"
    with tarfile.open(fileobj=io.BytesIO(raw), mode=mode) as tar:
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
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _find_data_dir(workdir: Path) -> Path:
    candidates = [
        workdir / "home" / "data",
        workdir / "data",
        workdir,
    ]
    for path in candidates:
        if path.exists() and list(path.rglob("*.csv")):
            return path
    return workdir


def _read_csv_any(path: Path) -> pd.DataFrame | None:
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_numeric_like_object(series: pd.Series, sample_size: int = 200) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return False
    ratio = sample.str.fullmatch(r"-?\d+(\.\d+)?").mean()
    return bool(ratio >= 0.8)


def _string_profile_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}__len"] = s.str.len()
    out[f"{prefix}__word_count"] = s.str.split().str.len().fillna(0)
    out[f"{prefix}__digit_count"] = s.str.count(r"\d")
    out[f"{prefix}__alpha_count"] = s.str.count(r"[A-Za-zА-Яа-я]")
    out[f"{prefix}__has_digit"] = s.str.contains(r"\d", regex=True).astype(int)
    out[f"{prefix}__is_missing_like"] = s.isin(["", "nan", "None", "none", "NaN"]).astype(int)
    out[f"{prefix}__unique_char_count"] = s.apply(lambda x: len(set(x)) if isinstance(x, str) else 0)
    out[f"{prefix}__is_all_caps"] = s.str.fullmatch(r"[A-Z]+").fillna(False).astype(int)
    first_token = s.str.split().str[0].fillna("")
    last_token = s.str.split().str[-1].fillna("")
    out[f"{prefix}__first_token"] = first_token
    out[f"{prefix}__last_token"] = last_token
    out[f"{prefix}__first_token_freq"] = first_token.map(first_token.value_counts(dropna=False))
    out[f"{prefix}__last_token_freq"] = last_token.map(last_token.value_counts(dropna=False))
    return out


def _infer_bool_like_columns(df: pd.DataFrame) -> list[str]:
    bool_like = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            bool_like.append(col)
            continue

        vals = (
            s.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        if 0 < len(vals) <= 5 and set(vals).issubset(
            {"true", "false", "0", "1", "yes", "no", "y", "n", "t", "f"}
        ):
            bool_like.append(col)
    return bool_like


def _normalize_bool_like(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "t": 1,
        "f": 0,
    }
    return lowered.map(mapping)


def _looks_like_id_name(name: str) -> bool:
    lower = name.strip().lower()
    if lower in {"id", "uid", "uuid", "guid"}:
        return True
    if lower.endswith("_id") or lower.endswith("id"):
        return True
    return any(token in lower for token in ["recordid", "rowid"])


def _is_id_like_series(series: pd.Series, *, unique_ratio_threshold: float = 0.9) -> bool:
    non_na = series.dropna()
    if len(non_na) == 0:
        return False

    unique_ratio = float(non_na.nunique(dropna=True)) / float(len(non_na))
    if unique_ratio < unique_ratio_threshold:
        return False

    if pd.api.types.is_numeric_dtype(series):
        return True

    as_str = non_na.astype(str)
    avg_len = float(as_str.str.len().mean()) if len(as_str) else 0.0
    return avg_len >= 6.0


def _candidate_structured_string_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        ss = s.dropna().astype(str).head(500)
        if len(ss) == 0:
            continue
        for sep in _SEPARATORS:
            ratio = ss.str.contains(re.escape(sep), regex=True).mean()
            if ratio >= 0.6:
                candidates.append(col)
                break
    return candidates


def _split_structured_column(series: pd.Series, col: str) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    out = pd.DataFrame(index=series.index)

    for sep in _SEPARATORS:
        sample = s.head(500)
        ratio = sample.str.contains(re.escape(sep), regex=True).mean()
        if ratio < 0.6:
            continue

        parts = s.str.split(sep, expand=True)
        if parts.shape[1] < 2:
            continue

        base = f"{col}__split"
        out[f"{base}_part0"] = parts[0]
        out[f"{base}_part0_freq"] = parts[0].map(parts[0].value_counts(dropna=False))

        part1_num = pd.to_numeric(parts[1], errors="coerce")
        if part1_num.notna().mean() >= 0.5:
            out[f"{base}_part1_num"] = part1_num
        else:
            out[f"{base}_part1"] = parts[1]
            out[f"{base}_part1_freq"] = parts[1].map(parts[1].value_counts(dropna=False))

        if parts.shape[1] >= 3:
            part_last = parts[parts.shape[1] - 1]
            out[f"{base}_last"] = part_last
            out[f"{base}_last_freq"] = part_last.map(part_last.value_counts(dropna=False))
            out[f"{base}_part0_last"] = parts[0].astype(str) + "__" + part_last.astype(str)

        group_key = parts[0].astype(str)
        out[f"{base}_group_size"] = group_key.map(group_key.value_counts(dropna=False))
        break

    return out


def _infer_positive_numeric_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    selected = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_na = s.dropna()
        if len(non_na) == 0:
            continue
        if (non_na >= 0).mean() >= 0.9:
            selected.append(col)
    return selected[:max_cols]


def _make_numeric_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    positive_numeric = _infer_positive_numeric_columns(df)
    if not positive_numeric:
        return out

    num_frame = df[positive_numeric].apply(pd.to_numeric, errors="coerce")
    out["__num_sum"] = num_frame.sum(axis=1)
    out["__num_mean"] = num_frame.mean(axis=1)
    out["__num_std"] = num_frame.std(axis=1)
    out["__num_max"] = num_frame.max(axis=1)
    out["__num_min"] = num_frame.min(axis=1)
    out["__num_missing_count"] = num_frame.isna().sum(axis=1)
    out["__num_zero_count"] = num_frame.fillna(0).eq(0).sum(axis=1)
    out["__num_positive_count"] = num_frame.fillna(0).gt(0).sum(axis=1)
    out["__num_log1p_sum"] = np.log1p(num_frame.clip(lower=0)).sum(axis=1)
    return out


def _make_group_aggregates(
    X: pd.DataFrame,
    raw_df: pd.DataFrame,
    max_group_keys: int = 3,
    max_numeric_targets: int = 6,
) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    structured_cols = _candidate_structured_string_columns(raw_df)
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()[:max_numeric_targets]

    used = 0
    for col in structured_cols:
        if used >= max_group_keys:
            break

        s = raw_df[col].fillna("").astype(str)
        found = False

        for sep in _SEPARATORS:
            ratio = s.head(500).str.contains(re.escape(sep), regex=True).mean()
            if ratio < 0.6:
                continue

            parts = s.str.split(sep, expand=True)
            if parts.shape[1] < 2:
                continue

            key = parts[0].astype(str)
            key_name = f"{col}__groupkey"
            out[key_name] = key
            out[f"{key_name}__size"] = key.map(key.value_counts(dropna=False))

            for num_col in numeric_cols:
                probe = pd.DataFrame({"g": key, "v": pd.to_numeric(X[num_col], errors="coerce")})
                grp = probe.groupby("g")["v"].agg(["mean", "max", "min"])
                out[f"{key_name}__{num_col}__mean"] = key.map(grp["mean"])
                out[f"{key_name}__{num_col}__max"] = key.map(grp["max"])
                out[f"{key_name}__{num_col}__min"] = key.map(grp["min"])

            used += 1
            found = True
            break

        if found and used >= max_group_keys:
            break

    return out


def _make_bool_numeric_interactions(
    X: pd.DataFrame,
    max_bool_cols: int = 4,
    max_num_cols: int = 6,
) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    bool_like_cols = []

    for col in X.columns:
        s = X[col]
        vals = (
            s.dropna().astype(str).str.strip().str.lower().unique().tolist()
            if s.dtype == object
            else []
        )
        if pd.api.types.is_bool_dtype(s):
            bool_like_cols.append(col)
        elif 0 < len(vals) <= 5 and set(vals).issubset({"true", "false", "0", "1", "yes", "no", "y", "n", "t", "f"}):
            bool_like_cols.append(col)

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    bool_like_cols = bool_like_cols[:max_bool_cols]
    numeric_cols = numeric_cols[:max_num_cols]

    for b in bool_like_cols:
        b_num = _normalize_bool_like(X[b])
        if b_num.isna().all():
            continue

        for n in numeric_cols:
            n_ser = pd.to_numeric(X[n], errors="coerce")
            out[f"{b}__x__{n}__zero"] = ((b_num == 1) & (n_ser.fillna(0) == 0)).astype(int)
            out[f"{b}__x__{n}__missing"] = ((b_num == 1) & n_ser.isna()).astype(int)

    return out


def make_features(X: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if _is_numeric_like_object(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in _infer_bool_like_columns(X):
        mapped = _normalize_bool_like(X[col])
        if mapped.notna().mean() >= 0.8:
            X[col] = mapped

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    for col in numeric_cols[:20]:
        X[f"{col}__is_missing"] = pd.to_numeric(X[col], errors="coerce").isna().astype(int)

    for col in numeric_cols[:12]:
        s = pd.to_numeric(X[col], errors="coerce")
        non_na = s.dropna()
        if len(non_na) == 0:
            continue
        if (non_na >= 0).mean() >= 0.9:
            X[f"{col}__log1p"] = np.log1p(s.clip(lower=0))
        if non_na.nunique(dropna=True) >= 10:
            X[f"{col}__sq"] = s * s
            abs_q95 = float(np.nanquantile(np.abs(non_na), 0.95))
            if abs_q95 > 0:
                X[f"{col}__z_like"] = (s - float(non_na.median())) / abs_q95

    object_cols = [c for c in X.columns if X[c].dtype == object]
    for col in object_cols[:10]:
        prof = _string_profile_features(raw_df[col] if col in raw_df.columns else X[col], col)
        X = pd.concat([X, prof], axis=1)

    for col in _candidate_structured_string_columns(raw_df)[:5]:
        if col in raw_df.columns:
            X = pd.concat([X, _split_structured_column(raw_df[col], col)], axis=1)

    X = pd.concat([X, _make_numeric_aggregates(X)], axis=1)
    X = pd.concat([X, _make_group_aggregates(X, raw_df)], axis=1)
    X = pd.concat([X, _make_bool_numeric_interactions(X)], axis=1)

    # Drop raw identifier-like columns after extracting useful statistics.
    drop_cols: list[str] = []
    for col in X.columns:
        if col in raw_df.columns:
            if _looks_like_id_name(col) or _is_id_like_series(raw_df[col]):
                drop_cols.append(col)
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    X_test = X_test.copy()
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    extra_cols = [c for c in X_test.columns if c not in X_train.columns]
    if extra_cols:
        X_test = X_test.drop(columns=extra_cols)
    return X_test[X_train.columns]


@dataclass
class CandidateResult:
    name: str
    score: float
    model: Any


def infer_task(train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict[str, Any]:
    pred_col = sample_df.columns[1]
    id_col = sample_df.columns[0]
    if id_col not in test_df.columns:
        shared = [c for c in sample_df.columns if c in test_df.columns]
        if shared:
            id_col = shared[0]

    if pred_col in train_df.columns and pred_col not in test_df.columns:
        target_col = pred_col
    else:
        candidate_cols = [c for c in train_df.columns if c not in test_df.columns]
        if not candidate_cols:
            raise ValueError("Could not detect target column: no train-only columns found.")

        non_id_candidates = [c for c in candidate_cols if not _looks_like_id_name(c)]
        if len(non_id_candidates) == 1:
            target_col = non_id_candidates[0]
        elif len(candidate_cols) == 1:
            target_col = candidate_cols[0]
        else:
            ranked: list[tuple[float, str]] = []
            for c in (non_id_candidates or candidate_cols):
                s = train_df[c]
                uniq = float(s.nunique(dropna=True))
                ratio = uniq / max(1.0, float(len(s)))
                # Penalize id-like columns; prefer lower-cardinality targets.
                penalty = 1.0 if _looks_like_id_name(c) or _is_id_like_series(s) else 0.0
                ranked.append((ratio + penalty, c))
            ranked.sort(key=lambda item: item[0])
            target_col = ranked[0][1]

    y = train_df[target_col]
    pred_values = sample_df[pred_col].dropna().astype(str).str.strip().str.lower().unique().tolist()

    is_bool_sample = set(pred_values).issubset({"true", "false"})
    is_int_sample = set(pred_values).issubset({"0", "1"})

    if is_bool_sample or is_int_sample:
        task_type = "binary_classification"
    else:
        nunique = y.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(y) and nunique > 20:
            task_type = "regression"
        elif nunique <= 20:
            task_type = "classification"
        else:
            task_type = "regression"

    return {
        "target_col": target_col,
        "id_col": id_col,
        "pred_col": pred_col,
        "task_type": task_type,
        "is_bool_sample": is_bool_sample,
        "is_int_sample": is_int_sample,
    }


def prepare_target(y_raw: pd.Series, task_type: str) -> tuple[pd.Series, dict[str, Any]]:
    meta: dict[str, Any] = {"task_type": task_type}

    if task_type in {"binary_classification", "classification"}:
        lowered = y_raw.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            y = lowered.map({"true": 1, "false": 0}).astype(int)
            meta["original_format"] = "bool_str"
            meta["task_type"] = "binary_classification"
            return y, meta

        if pd.api.types.is_bool_dtype(y_raw):
            y = y_raw.astype(int)
            meta["original_format"] = "bool"
            meta["task_type"] = "binary_classification"
            return y, meta

        nunique = y_raw.nunique(dropna=True)
        if nunique == 2:
            classes = sorted(pd.Series(y_raw.dropna().unique()).tolist(), key=lambda x: str(x))
            mapping = {classes[0]: 0, classes[1]: 1}
            y = y_raw.map(mapping).astype(int)
            meta["original_format"] = "binary_generic"
            meta["inverse_mapping"] = {0: classes[0], 1: classes[1]}
            meta["task_type"] = "binary_classification"
            return y, meta

        classes = sorted(pd.Series(y_raw.dropna().unique()).tolist(), key=lambda x: str(x))
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).astype(int)
        meta["original_format"] = "multiclass"
        meta["inverse_mapping"] = {i: c for i, c in enumerate(classes)}
        meta["task_type"] = "classification"
        return y, meta

    y = pd.to_numeric(y_raw, errors="coerce")
    if y.isna().any():
        y = y.fillna(y.median())
    meta["original_format"] = "regression"
    meta["task_type"] = "regression"
    return y, meta


def _build_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre_ohe = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    pre_ord = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    return pre_ohe, pre_ord


def build_candidates(X: pd.DataFrame, task_type: str) -> list[tuple[str, Any]]:
    pre_ohe, pre_ord = _build_preprocessors(X)
    candidates: list[tuple[str, Any]] = []

    if task_type == "binary_classification":
        candidates.extend(
            [
                (
                    "logreg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", LogisticRegression(max_iter=5000, C=1.4, solver="liblinear", random_state=42)),
                        ]
                    ),
                ),
                (
                    "extratrees_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", ExtraTreesClassifier(n_estimators=1200, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "rf_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", RandomForestClassifier(n_estimators=900, max_depth=18, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "hgb_ordinal_main",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            ("model", HistGradientBoostingClassifier(learning_rate=0.04, max_depth=8, max_iter=500, l2_regularization=0.05, random_state=42)),
                        ]
                    ),
                ),
                (
                    "hgb_ordinal_alt",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            ("model", HistGradientBoostingClassifier(learning_rate=0.03, max_depth=10, max_iter=700, l2_regularization=0.08, random_state=42)),
                        ]
                    ),
                ),
            ]
        )
    elif task_type == "classification":
        candidates.extend(
            [
                ("logreg_ohe", Pipeline([("pre", pre_ohe), ("model", LogisticRegression(max_iter=4000, random_state=42))])),
                ("extratrees_ohe", Pipeline([("pre", pre_ohe), ("model", ExtraTreesClassifier(n_estimators=1000, random_state=42, n_jobs=-1))])),
                ("hgb_ordinal", Pipeline([("pre", pre_ord), ("model", HistGradientBoostingClassifier(max_iter=500, random_state=42))])),
            ]
        )
    else:
        candidates.extend(
            [
                ("ridge_ohe", Pipeline([("pre", pre_ohe), ("model", Ridge(alpha=1.0, random_state=42))])),
                ("extratrees_reg_ohe", Pipeline([("pre", pre_ohe), ("model", ExtraTreesRegressor(n_estimators=1000, random_state=42, n_jobs=-1))])),
                ("rf_reg_ohe", Pipeline([("pre", pre_ohe), ("model", RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1))])),
                ("hgb_reg_ordinal", Pipeline([("pre", pre_ord), ("model", HistGradientBoostingRegressor(max_iter=500, random_state=42))])),
            ]
        )

    return candidates


def evaluate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: list[tuple[str, Any]],
    task_type: str,
    logs: list[str],
) -> list[CandidateResult]:
    results: list[CandidateResult] = []

    if task_type in {"binary_classification", "classification"}:
        cv = StratifiedKFold(n_splits=5 if len(y) >= 500 else 4, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=5 if len(y) >= 500 else 4, shuffle=True, random_state=42)
        scoring = make_scorer(lambda yt, yp: -float(np.sqrt(mean_squared_error(yt, yp))), greater_is_better=True)

    for name, model in candidates:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            score = float(np.mean(scores))
            results.append(CandidateResult(name=name, score=score, model=model))
            logs.append(f"{name}: {score:.6f}")
        except Exception as e:
            logs.append(f"{name}: FAILED ({e})")

    if not results:
        raise RuntimeError("All candidate models failed.")

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _best_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_grid: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_acc = -1.0
    for threshold in threshold_grid:
        acc = float(accuracy_score(y_true, (probabilities >= threshold).astype(int)))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


def _predict_binary_with_blend(
    ranked_results: list[CandidateResult],
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logs: list[str],
) -> np.ndarray:
    _ = train_df
    _ = test_df
    blend_members = [r for r in ranked_results if hasattr(r.model, "predict_proba")][:3]
    if len(blend_members) < 2:
        best = clone(ranked_results[0].model)
        best.fit(X_train, y)
        logs.append(f"binary_fallback_model: {ranked_results[0].name}")
        if hasattr(best, "predict_proba"):
            return (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        return best.predict(X_test)

    raw_weights = np.array([max(1e-6, float(r.score)) ** 2 for r in blend_members], dtype=float)
    weight_sum = raw_weights.sum()
    if weight_sum <= 0:
        raw_weights = np.ones_like(raw_weights)
    else:
        raw_weights = raw_weights / weight_sum

    for member, w in zip(blend_members, raw_weights):
        logs.append(f"binary_blend_member: {member.name} weight={w:.4f}")

    cv = StratifiedKFold(n_splits=5 if len(y) >= 500 else 4, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y), dtype=float)
    oof_seen = np.zeros(len(y), dtype=bool)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y), start=1):
        fold_preds: list[np.ndarray] = []
        fold_weights: list[float] = []
        for member, weight in zip(blend_members, raw_weights):
            try:
                model = clone(member.model)
                model.fit(X_train.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
                fold_preds.append(pred)
                fold_weights.append(float(weight))
            except Exception as e:
                logs.append(f"binary_blend_fold_{fold}_failed: {member.name} ({e})")

        if not fold_preds:
            continue

        fold_matrix = np.column_stack(fold_preds)
        oof_proba[valid_idx] = np.average(
            fold_matrix,
            axis=1,
            weights=np.asarray(fold_weights, dtype=float),
        )
        oof_seen[valid_idx] = True

    if int(oof_seen.sum()) >= max(10, int(0.8 * len(y))):
        threshold_grid = np.linspace(0.35, 0.65, 61)
        threshold, oof_acc = _best_threshold(
            probabilities=oof_proba[oof_seen],
            y_true=y.iloc[oof_seen].to_numpy(),
            threshold_grid=threshold_grid,
        )
        logs.append(f"binary_blend_threshold: {threshold:.4f} oof_acc={oof_acc:.6f}")
    else:
        threshold = 0.5
        logs.append("binary_blend_threshold: default=0.5000 (insufficient_oof)")

    test_preds: list[np.ndarray] = []
    test_weights: list[float] = []
    for member, weight in zip(blend_members, raw_weights):
        try:
            model = clone(member.model)
            model.fit(X_train, y)
            test_preds.append(model.predict_proba(X_test)[:, 1])
            test_weights.append(float(weight))
        except Exception as e:
            logs.append(f"binary_blend_full_fit_failed: {member.name} ({e})")

    if not test_preds:
        best = clone(ranked_results[0].model)
        best.fit(X_train, y)
        logs.append(f"binary_fallback_model_after_blend_fail: {ranked_results[0].name}")
        if hasattr(best, "predict_proba"):
            return (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        return best.predict(X_test)

    blended_test_proba = np.average(
        np.column_stack(test_preds),
        axis=1,
        weights=np.asarray(test_weights, dtype=float),
    )
    return (blended_test_proba >= threshold).astype(int)


def format_predictions(
    preds: np.ndarray,
    y_raw: pd.Series,
    sample_df: pd.DataFrame,
    pred_col: str,
    task_meta: dict[str, Any],
) -> pd.Series:
    preds_series = pd.Series(preds)
    sample_vals = sample_df[pred_col].dropna().astype(str).str.strip().str.lower().unique().tolist()

    if task_meta["task_type"] == "regression":
        return preds_series

    if set(sample_vals).issubset({"true", "false"}):
        return preds_series.astype(int).map({1: "True", 0: "False"}).fillna("False")

    if set(sample_vals).issubset({"0", "1"}):
        return preds_series.astype(int)

    inverse_mapping = task_meta.get("inverse_mapping")
    if inverse_mapping is not None:
        return preds_series.map(inverse_mapping)

    unique_raw = set(y_raw.dropna().astype(str).str.strip().str.lower().unique().tolist())
    if unique_raw.issubset({"true", "false"}):
        return preds_series.astype(int).map({1: "True", 0: "False"}).fillna("False")

    return preds_series


def sanitize_submission(submission: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    fixed = sample_df.copy()
    pred_col = fixed.columns[1]
    fixed = fixed.astype({pred_col: "object"})

    sample_vals = (
        sample_df.iloc[:, 1]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if submission.shape[1] >= 2:
        n = min(len(submission), len(fixed))
        fixed.iloc[:n, 0] = submission.iloc[:n, 0].values
        fixed.iloc[:n, 1] = submission.iloc[:n, 1].values

    if set(sample_vals).issubset({"true", "false"}):
        fixed[pred_col] = (
            fixed[pred_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"1": "True", "0": "False", "true": "True", "false": "False"})
            .fillna("False")
        )
    elif set(sample_vals).issubset({"0", "1"}):
        fixed[pred_col] = (
            fixed[pred_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"1": 1, "0": 0, "true": 1, "false": 0})
            .fillna(0)
            .astype(int)
        )

    return fixed


def solve_competition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    task: dict[str, Any],
    logs: list[str],
) -> tuple[pd.DataFrame, str]:
    target_col = task["target_col"]
    id_col = task["id_col"]
    pred_col = task["pred_col"]
    task_type = task["task_type"]

    train_df = train_df.copy()
    test_df = test_df.copy()

    y_raw = train_df[target_col].copy()
    X_train_raw = train_df.drop(columns=[target_col], errors="ignore")
    X_test_raw = test_df.copy()

    if id_col in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=[id_col], errors="ignore")
    if id_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[id_col], errors="ignore")

    X_train = make_features(X_train_raw, train_df)
    X_test = make_features(X_test_raw, test_df)
    X_test = align_columns(X_train, X_test)

    y, task_meta = prepare_target(y_raw, task_type)
    effective_task_type = task_meta["task_type"]

    candidates = build_candidates(X_train, effective_task_type)
    ranked = evaluate_candidates(X_train, y, candidates, effective_task_type, logs=logs)
    best = ranked[0]

    if effective_task_type == "binary_classification":
        preds = _predict_binary_with_blend(
            ranked_results=ranked,
            X_train=X_train,
            y=y,
            X_test=X_test,
            train_df=train_df,
            test_df=test_df,
            logs=logs,
        )
    else:
        final_model = clone(best.model)
        final_model.fit(X_train, y)
        preds = final_model.predict(X_test)

    formatted_preds = format_predictions(
        preds=preds,
        y_raw=y_raw,
        sample_df=sample_df,
        pred_col=pred_col,
        task_meta=task_meta,
    )

    submission = pd.DataFrame(
        {
            sample_df.columns[0]: test_df[id_col].values if id_col in test_df.columns else sample_df.iloc[:, 0].values,
            sample_df.columns[1]: pd.Series(formatted_preds, dtype="object"),
        }
    )
    submission = sanitize_submission(submission, sample_df)

    summary = "\n".join(
        [
            f"target_col={target_col}",
            f"id_col={id_col}",
            f"pred_col={pred_col}",
            f"task_type={effective_task_type}",
            f"train_shape={train_df.shape}",
            f"test_shape={test_df.shape}",
            f"best_model={best.name}",
            f"best_cv={best.score:.6f}",
            "",
            "candidate_results:",
            *logs,
        ]
    )
    return submission, summary


class Agent:
    def __init__(self) -> None:
        self._done_context: set[str] = set()
        self.logs: list[str] = []
        self._current_work_dir: Path | None = None
        self._current_text: str = ""
        self._current_updater: TaskUpdater | None = None

    def _build_fallback_submission(self, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        fallback = sample_df.copy()
        id_col = sample_df.columns[0]

        if id_col in test_df.columns:
            n = min(len(fallback), len(test_df))
            fallback.iloc[:n, 0] = test_df.iloc[:n][id_col].values

        return sanitize_submission(fallback, sample_df)

    def _validate_submission(self, submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
        if list(submission_df.columns) != list(sample_df.columns):
            raise ValueError(f"Expected columns {list(sample_df.columns)}, got {list(submission_df.columns)}")
        if len(submission_df) != len(sample_df):
            raise ValueError(f"Expected {len(sample_df)} rows, got {len(submission_df)}")
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
        _ = description
        return solve_competition(
            train_df=train_df,
            test_df=test_df,
            sample_df=sample_df,
            task=task,
            logs=self.logs,
        )

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
        self._current_text = get_message_text(message)

        if ctx in self._done_context:
            logger.info("Context %s already finished; ack", ctx)
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            logger.error("No competition tar.gz in message")
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
            self._current_work_dir = work_dir
            self._current_updater = updater

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

            description = ""
            for candidate in ["description.md", "description.txt", "README.md"]:
                p = _find_first(data_dir, candidate)
                if p and p.exists():
                    description = p.read_text(encoding="utf-8", errors="replace")[:20000]
                    break

            train_path = _find_first(data_dir, "train.csv")
            test_path = _find_first(data_dir, "test.csv")
            sample_path = _find_first(data_dir, "sample_submission.csv")

            if train_path is None or test_path is None or sample_path is None:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: could not find train/test/sample_submission files"))],
                    name="Error",
                )
                return

            train_df = _read_csv_any(train_path)
            test_df = _read_csv_any(test_path)
            sample_df = _read_csv_any(sample_path)

            if train_df is None or test_df is None or sample_df is None:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: failed to read train/test/sample_submission files"))],
                    name="Error",
                )
                return

            self.logs.append(
                json.dumps(
                    {
                        "paths": {
                            "train": str(train_path),
                            "test": str(test_path),
                            "sample_submission": str(sample_path),
                        },
                        "train_shape": list(train_df.shape),
                        "test_shape": list(test_df.shape),
                        "sample_shape": list(sample_df.shape),
                    },
                    ensure_ascii=False,
                )
            )

            baseline_submission_df = self._build_fallback_submission(test_df=test_df, sample_df=sample_df)
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

            try:
                task = infer_task(train_df, test_df, sample_df)
                self.logs.append(json.dumps(task, ensure_ascii=False))

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("Building features and evaluating candidate models..."),
                )

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

            except Exception as exc:
                logger.exception("Modeling failed")
                self.logs.append(f"fallback_used: {exc}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "Model training failed or suspicious submission detected; using safe baseline submission."
                    ),
                )
            finally:
                self._current_work_dir = None
                self._current_updater = None
                self._current_text = ""

        self._done_context.add(ctx)
        logger.info("Finished context %s", ctx)
