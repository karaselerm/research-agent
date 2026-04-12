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
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import align_columns, make_features
from .submission import sanitize_submission


@dataclass
class CandidateResult:
    name: str
    score: float
    model: Any


@dataclass(frozen=True)
class BinaryPostprocessConfig:
    threshold: float = 0.5
    min_group_size: int | None = None
    high_mean_prob: float | None = None
    low_mean_prob: float | None = None


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
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
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
                            ("model", LogisticRegression(max_iter=4000, C=1.4, solver="liblinear", random_state=42)),
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
                                    n_estimators=900,
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
                    "extratrees_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    n_estimators=1200,
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
                    "hgb_ordinal_main",
                    Pipeline(
                        [
                            ("pre", pre_ord),
                            (
                                "model",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.04,
                                    max_depth=8,
                                    max_iter=500,
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
                                    max_iter=700,
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
                    "rf_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "extratrees_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", ExtraTreesClassifier(n_estimators=1000, random_state=42, n_jobs=-1)),
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
                    "rf_reg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1)),
                        ]
                    ),
                ),
                (
                    "extratrees_reg_ohe",
                    Pipeline(
                        [
                            ("pre", pre_ohe),
                            ("model", ExtraTreesRegressor(n_estimators=1000, random_state=42, n_jobs=-1)),
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

    if task_type == "binary_classification":
        top = results[:3]
        ensemble_estimators = []
        for i, r in enumerate(top):
            if hasattr(r.model, "predict_proba"):
                ensemble_estimators.append((f"m{i}", r.model))

        if len(ensemble_estimators) >= 2:
            try:
                ensemble = VotingClassifier(
                    estimators=ensemble_estimators,
                    voting="soft",
                    n_jobs=1,
                )
                cv = StratifiedKFold(
                    n_splits=5 if len(y) >= 500 else 4,
                    shuffle=True,
                    random_state=42,
                )
                ens_score = float(
                    np.mean(cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy", n_jobs=1))
                )
                logs.append(f"soft_voting_top_models: {ens_score:.6f}")
                if ens_score > results[0].score:
                    results.insert(
                        0,
                        CandidateResult(
                            name="soft_voting_top_models",
                            score=ens_score,
                            model=ensemble,
                        ),
                    )
            except Exception as e:
                logs.append(f"soft_voting_top_models: FAILED ({e})")

    return results


def _apply_group_consensus(
    probabilities: np.ndarray,
    group_values: pd.Series,
    config: BinaryPostprocessConfig,
    logs: list[str] | None = None,
    log_prefix: str = "group_consensus",
) -> np.ndarray:
    if (
        config.min_group_size is None
        or config.high_mean_prob is None
        or config.low_mean_prob is None
    ):
        return probabilities

    if len(group_values) != len(probabilities):
        if logs is not None:
            logs.append(f"{log_prefix}: skipped (group/value length mismatch)")
        return probabilities

    out = probabilities.copy()
    group_id = group_values.astype(str).str.split("_", n=1).str[0]
    probe = pd.DataFrame({"g": group_id, "p": out})
    stats = probe.groupby("g")["p"].agg(["mean", "count"])
    mean_map = probe["g"].map(stats["mean"])
    count_map = probe["g"].map(stats["count"])

    strong_high = (
        (count_map >= config.min_group_size)
        & (mean_map >= config.high_mean_prob)
    )
    strong_low = (
        (count_map >= config.min_group_size)
        & (mean_map <= config.low_mean_prob)
    )
    out[strong_high.values] = np.maximum(out[strong_high.values], mean_map[strong_high].to_numpy())
    out[strong_low.values] = np.minimum(out[strong_low.values], mean_map[strong_low].to_numpy())

    changed = int(np.sum(strong_high.values) + np.sum(strong_low.values))
    if logs is not None:
        logs.append(
            (
                f"{log_prefix}: adjusted_rows={changed} "
                f"min_group_size={config.min_group_size} "
                f"high_mean_prob={config.high_mean_prob:.3f} "
                f"low_mean_prob={config.low_mean_prob:.3f}"
            )
        )
    return out


def _best_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_grid: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_acc = -1.0
    for t in threshold_grid:
        acc = float(accuracy_score(y_true, (probabilities >= t).astype(int)))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(t)
    return best_threshold, best_acc


def _tune_group_postprocess(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    group_values: pd.Series,
    base_threshold: float,
    base_accuracy: float,
) -> tuple[BinaryPostprocessConfig, float]:
    best_cfg = BinaryPostprocessConfig(threshold=base_threshold)
    best_acc = base_accuracy

    threshold_grid = np.linspace(0.42, 0.58, 65)
    for min_group_size in (2, 3):
        for high_mean_prob in (0.85, 0.875, 0.9):
            for low_mean_prob in (0.25, 0.275, 0.3):
                cfg = BinaryPostprocessConfig(
                    threshold=base_threshold,
                    min_group_size=min_group_size,
                    high_mean_prob=high_mean_prob,
                    low_mean_prob=low_mean_prob,
                )
                adjusted = _apply_group_consensus(
                    probabilities=probabilities,
                    group_values=group_values,
                    config=cfg,
                )
                threshold, acc = _best_threshold(adjusted, y_true, threshold_grid)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = BinaryPostprocessConfig(
                        threshold=threshold,
                        min_group_size=min_group_size,
                        high_mean_prob=high_mean_prob,
                        low_mean_prob=low_mean_prob,
                    )

    return best_cfg, best_acc


def _calibrate_binary_postprocess(
    model: Any,
    X_train: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    logs: list[str],
) -> BinaryPostprocessConfig:
    if not hasattr(model, "predict_proba"):
        return BinaryPostprocessConfig()

    try:
        idx = np.arange(len(y))
        X_fit, X_val, y_fit, y_val, _, idx_val = train_test_split(
            X_train,
            y,
            idx,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model_for_calibration = clone(model)
        model_for_calibration.fit(X_fit, y_fit)
        val_probabilities = model_for_calibration.predict_proba(X_val)[:, 1]

        threshold_grid = np.linspace(0.42, 0.58, 65)
        threshold, base_acc = _best_threshold(
            probabilities=val_probabilities,
            y_true=np.asarray(y_val),
            threshold_grid=threshold_grid,
        )
        best_cfg = BinaryPostprocessConfig(threshold=threshold)
        best_acc = base_acc

        logs.append(
            f"binary_threshold_calibration: threshold={threshold:.4f} accuracy={base_acc:.6f}"
        )

        if "PassengerId" in train_df.columns:
            group_values = train_df.iloc[idx_val]["PassengerId"]
            tuned_cfg, tuned_acc = _tune_group_postprocess(
                probabilities=val_probabilities,
                y_true=np.asarray(y_val),
                group_values=group_values,
                base_threshold=threshold,
                base_accuracy=base_acc,
            )
            if tuned_acc > best_acc:
                best_cfg = tuned_cfg
                best_acc = tuned_acc
                logs.append(
                    (
                        "binary_group_calibration: improved "
                        f"accuracy={tuned_acc:.6f} threshold={tuned_cfg.threshold:.4f} "
                        f"min_group_size={tuned_cfg.min_group_size} "
                        f"high={tuned_cfg.high_mean_prob:.3f} low={tuned_cfg.low_mean_prob:.3f}"
                    )
                )
            else:
                logs.append("binary_group_calibration: no_improvement")

        return best_cfg
    except Exception as e:
        logs.append(f"binary_postprocess_calibration_failed: {e}")
        return BinaryPostprocessConfig()


def _predict_binary_with_best_model(
    ranked_results: list[CandidateResult],
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logs: list[str],
) -> np.ndarray:
    best = ranked_results[0]
    model = best.model
    postprocess_cfg = _calibrate_binary_postprocess(
        model=model,
        X_train=X_train,
        y=y,
        train_df=train_df,
        logs=logs,
    )

    model.fit(X_train, y)
    if not hasattr(model, "predict_proba"):
        logs.append(f"binary_inference: model={best.name} uses direct predict (no probability)")
        return model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]
    if (
        "PassengerId" in test_df.columns
        and postprocess_cfg.min_group_size is not None
    ):
        probabilities = _apply_group_consensus(
            probabilities=probabilities,
            group_values=test_df["PassengerId"],
            config=postprocess_cfg,
            logs=logs,
            log_prefix="group_consensus_inference",
        )

    logs.append(
        (
            f"binary_inference: model={best.name} "
            f"threshold={postprocess_cfg.threshold:.4f}"
        )
    )
    return (probabilities >= postprocess_cfg.threshold).astype(int)


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
        preds = _predict_binary_with_best_model(
            ranked_results=ranked,
            X_train=X_train,
            y=y,
            X_test=X_test,
            train_df=train_df,
            test_df=test_df,
            logs=logs,
        )
    else:
        final_model = best.model
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
