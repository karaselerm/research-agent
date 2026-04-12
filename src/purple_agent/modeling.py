from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import align_columns, make_features
from .submission import sanitize_submission


@dataclass
class CandidateResult:
    name: str
    score: float
    model: Any


def infer_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> dict[str, Any]:
    pred_col = sample_df.columns[1]
    id_col = sample_df.columns[0]

    if pred_col in train_df.columns:
        target_col = pred_col
    else:
        candidate_cols = [c for c in train_df.columns if c not in test_df.columns]
        if len(candidate_cols) == 1:
            target_col = candidate_cols[0]
        else:
            target_col = None
            for c in ["target", "label", "Transported", "Survived", "Response", "Outcome"]:
                if c in train_df.columns:
                    target_col = c
                    break
            if target_col is None:
                raise ValueError("Could not detect target column.")

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
            meta["class_labels"] = [0, 1]
            meta["task_type"] = "binary_classification"
            return y, meta

        if pd.api.types.is_bool_dtype(y_raw):
            y = y_raw.astype(int)
            meta["original_format"] = "bool"
            meta["class_labels"] = [0, 1]
            meta["task_type"] = "binary_classification"
            return y, meta

        nunique = y_raw.nunique(dropna=True)
        if nunique == 2:
            classes = sorted(pd.Series(y_raw.dropna().unique()).tolist(), key=lambda x: str(x))
            mapping = {classes[0]: 0, classes[1]: 1}
            y = y_raw.map(mapping).astype(int)
            meta["original_format"] = "binary_generic"
            meta["inverse_mapping"] = {0: classes[0], 1: classes[1]}
            meta["class_labels"] = [0, 1]
            meta["task_type"] = "binary_classification"
            return y, meta

        classes = sorted(pd.Series(y_raw.dropna().unique()).tolist(), key=lambda x: str(x))
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).astype(int)
        meta["original_format"] = "multiclass"
        meta["inverse_mapping"] = {i: c for i, c in enumerate(classes)}
        meta["class_labels"] = list(range(len(classes)))
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
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
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


def build_candidates(
    X: pd.DataFrame,
    task_type: str,
) -> list[tuple[str, Any]]:
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
                            ("model", LogisticRegression(max_iter=4000, C=1.2, solver="liblinear", random_state=42)),
                        ]
                    ),
                ),
                (
                    "extratrees_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    n_estimators=900,
                                    max_depth=None,
                                    min_samples_leaf=1,
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "rf_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            (
                                "model",
                                RandomForestClassifier(
                                    n_estimators=700,
                                    max_depth=18,
                                    min_samples_leaf=2,
                                    max_features="sqrt",
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "hgb_ordinal_main",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            (
                                "model",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.04,
                                    max_depth=8,
                                    max_iter=450,
                                    l2_regularization=0.05,
                                    random_state=42,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "hgb_ordinal_alt",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            (
                                "model",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.03,
                                    max_depth=10,
                                    max_iter=650,
                                    l2_regularization=0.08,
                                    random_state=42,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
    elif task_type == "classification":
        candidates.extend(
            [
                (
                    "logreg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", LogisticRegression(max_iter=3000, random_state=42)),
                        ]
                    ),
                ),
                (
                    "extratrees_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", ExtraTreesClassifier(n_estimators=800, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "hgb_ordinal",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            ("model", HistGradientBoostingClassifier(max_iter=450, random_state=42)),
                        ]
                    ),
                ),
            ]
        )
    else:
        candidates.extend(
            [
                (
                    "ridge_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", Ridge(alpha=1.0, random_state=42)),
                        ]
                    ),
                ),
                (
                    "extratrees_reg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", ExtraTreesRegressor(n_estimators=800, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "rf_reg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "hgb_reg_ordinal",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            ("model", HistGradientBoostingRegressor(max_iter=450, random_state=42)),
                        ]
                    ),
                ),
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
        n_splits = 5 if len(y) >= 500 else 4
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        n_splits = 5 if len(y) >= 500 else 4
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = make_scorer(
            lambda yt, yp: -float(np.sqrt(mean_squared_error(yt, yp))),
            greater_is_better=True,
        )

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


def _apply_soft_group_consensus(
    probabilities: np.ndarray,
    group_values: pd.Series,
    logs: list[str] | None = None,
    log_prefix: str = "soft_group_consensus",
) -> np.ndarray:
    if len(group_values) != len(probabilities):
        if logs is not None:
            logs.append(f"{log_prefix}: skipped (length mismatch)")
        return probabilities

    out = probabilities.copy()
    group_id = group_values.astype(str).str.split("_", n=1).str[0]

    probe = pd.DataFrame({"g": group_id, "p": out})
    stats = probe.groupby("g")["p"].agg(["mean", "count"])

    mean_map = probe["g"].map(stats["mean"]).to_numpy()
    count_map = probe["g"].map(stats["count"]).to_numpy()

    strong_high = (count_map >= 2) & (mean_map >= 0.88)
    strong_low = (count_map >= 2) & (mean_map <= 0.12)

    out[strong_high] = np.maximum(out[strong_high], 0.90)
    out[strong_low] = np.minimum(out[strong_low], 0.10)

    if logs is not None:
        changed = int(strong_high.sum() + strong_low.sum())
        logs.append(f"{log_prefix}: adjusted_rows={changed}")

    return out


def _predict_binary_with_blend(
    ranked_results: list[CandidateResult],
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    logs: list[str],
) -> np.ndarray:
    top = ranked_results[:4]

    proba_preds: list[np.ndarray] = []
    weights: list[float] = []

    for r in top:
        model = clone(r.model)
        try:
            model.fit(X_train, y)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
                p = np.asarray(pred, dtype=float)

            proba_preds.append(p)
            weight = max(1e-6, float(r.score)) ** 2
            weights.append(weight)
            logs.append(f"blend_member: {r.name} weight={weight:.6f}")
        except Exception as e:
            logs.append(f"blend_member_failed: {r.name} ({e})")

    if not proba_preds:
        best = clone(ranked_results[0].model)
        best.fit(X_train, y)
        if hasattr(best, "predict_proba"):
            return (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        return best.predict(X_test)

    matrix = np.column_stack(proba_preds)
    blend_proba = np.average(matrix, axis=1, weights=np.array(weights))

    # generic structural smoothing:
    # uses repeated composite ID prefixes only if present
    if test_df.shape[1] > 0:
        for col in test_df.columns:
            s = test_df[col]
            if s.dtype == object:
                sample = s.dropna().astype(str).head(300)
                if len(sample) > 0 and sample.str.contains("_").mean() >= 0.6:
                    blend_proba = _apply_soft_group_consensus(
                        probabilities=blend_proba,
                        group_values=test_df[col],
                        logs=logs,
                        log_prefix=f"soft_group_consensus_{col}",
                    )
                    break

    return (blend_proba >= 0.5).astype(int)


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

    original_format = task_meta.get("original_format")
    inverse_mapping = task_meta.get("inverse_mapping")

    if inverse_mapping is not None:
        return preds_series.map(inverse_mapping)

    if original_format in {"bool", "bool_str", "binary_generic"}:
        unique_raw = set(y_raw.dropna().astype(str).str.strip().str.lower().unique().tolist())
        if unique_raw.issubset({"true", "false"}):
            return preds_series.astype(int).map({1: "True", 0: "False"}).fillna("False")

    return preds_series


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
            sample_df.columns[0]: test_df[id_col].values
            if id_col in test_df.columns
            else sample_df.iloc[:, 0].values,
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