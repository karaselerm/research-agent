from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


_SEPARATORS = ["_", "/", "-", "|", ":"]


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

    first_token = s.str.split().str[0]
    last_token = s.str.split().str[-1]

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
    mapped = lowered.map(mapping)
    return mapped


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
        non_negative_ratio = (non_na >= 0).mean()
        if non_negative_ratio >= 0.9:
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


def _make_bool_numeric_interactions(X: pd.DataFrame, max_bool_cols: int = 4, max_num_cols: int = 6) -> pd.DataFrame:
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


def make_features(
    X: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    X = X.copy()

    # 1) convert numeric-like object columns
    for col in X.columns:
        if _is_numeric_like_object(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 2) normalize obvious bool-like columns
    for col in _infer_bool_like_columns(X):
        mapped = _normalize_bool_like(X[col])
        if mapped.notna().mean() >= 0.8:
            X[col] = mapped

    # 3) generic missing indicators for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    for col in numeric_cols[:20]:
        X[f"{col}__is_missing"] = pd.to_numeric(X[col], errors="coerce").isna().astype(int)

    # 4) log1p features for skewed non-negative numerics
    for col in numeric_cols[:12]:
        s = pd.to_numeric(X[col], errors="coerce")
        non_na = s.dropna()
        if len(non_na) == 0:
            continue
        if (non_na >= 0).mean() >= 0.9:
            X[f"{col}__log1p"] = np.log1p(s.clip(lower=0))

    # 5) generic string profile features
    object_cols = [c for c in X.columns if X[c].dtype == object]
    for col in object_cols[:10]:
        prof = _string_profile_features(raw_df[col] if col in raw_df.columns else X[col], col)
        X = pd.concat([X, prof], axis=1)

    # 6) structured split features
    for col in _candidate_structured_string_columns(raw_df)[:5]:
        if col in raw_df.columns:
            split_feats = _split_structured_column(raw_df[col], col)
            X = pd.concat([X, split_feats], axis=1)

    # 7) generic numeric aggregates
    X = pd.concat([X, _make_numeric_aggregates(X)], axis=1)

    # 8) group aggregates from repeated structured prefixes
    X = pd.concat([X, _make_group_aggregates(X, raw_df)], axis=1)

    # 9) bool x numeric interactions
    X = pd.concat([X, _make_bool_numeric_interactions(X)], axis=1)

    # 10) replace inf
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