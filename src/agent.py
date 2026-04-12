import base64
import io
import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message


@dataclass
class CandidateResult:
    name: str
    score: float
    model: Any


class Agent:
    def __init__(self):
        self.logs: list[str] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reading competition bundle..."),
        )

        with tempfile.TemporaryDirectory(prefix="purple-agent-") as tmpdir:
            workdir = Path(tmpdir)
            data_dir = self._extract_competition_bundle(message, workdir)

            description = self._read_description(data_dir)
            train_df, test_df, sample_df, paths_info = self._load_core_files(data_dir)

            # # await updater.add_artifact(
            #     parts=[
            #         Part(
            #             root=TextPart(
            #                 text=json.dumps(
            #                     {
            #                         "data_dir": str(data_dir),
            #                         "paths": paths_info,
            #                         "train_shape": list(train_df.shape),
            #                         "test_shape": list(test_df.shape),
            #                         "sample_shape": list(sample_df.shape),
            #                     },
            #                     ensure_ascii=False,
            #                     indent=2,
            #                 )
            #             )
            #         )
            #     ],
            #     name="debug_dataset_detection.txt",
            # )

            # Submit a safe baseline early so the evaluator always receives
            # a valid submission artifact even if later model training times out.
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

            task = self._infer_task(train_df, test_df, sample_df)
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

                # await updater.add_artifact(
                #     parts=[Part(root=TextPart(text=summary))],
                #     name="debug_report.txt",
                # )

                # await updater.add_artifact(
                #     parts=[Part(root=TextPart(text=submission_df.head(20).to_csv(index=False)))],
                #     name="debug_submission_preview.csv",
                # )
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
                    new_agent_text_message("Model training failed or timed out risk detected; using safe baseline submission."),
                )


    def _extract_competition_bundle(self, message: Message, workdir: Path) -> Path:
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
            tar.extractall(extract_root)

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

    def _read_description(self, data_dir: Path) -> str:
        for name in ["description.md", "description.txt", "README.md"]:
            p = data_dir / name
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")[:20000]
        return ""

    def _load_core_files(
        self, data_dir: Path
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
                df = self._read_csv_any(p)
                if df is not None and df.shape[1] == 2:
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

        train_df = self._read_csv_any(train_path)
        test_df = self._read_csv_any(test_path)
        sample_df = self._read_csv_any(sample_path)

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

    def _read_csv_any(self, path: Path) -> pd.DataFrame | None:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return None

    def _infer_task(
        self,
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

    def _solve_competition(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        description: str,
        task: dict[str, Any],
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

        X_train = self._make_features(X_train_raw, train_df, description)
        X_test = self._make_features(X_test_raw, test_df, description)
        X_test = self._align_columns(X_train, X_test)

        y, task_meta = self._prepare_target(y_raw, task_type)
        effective_task_type = task_meta["task_type"]

        candidates = self._build_candidates(X_train, y, effective_task_type)
        best = self._evaluate_candidates(X_train, y, candidates, effective_task_type)

        final_model = best.model
        final_model.fit(X_train, y)

        if effective_task_type in {"binary_classification", "classification"}:
            preds = final_model.predict(X_test)
        else:
            preds = final_model.predict(X_test)

        formatted_preds = self._format_predictions(
            preds=preds,
            y_raw=y_raw,
            sample_df=sample_df,
            pred_col=pred_col,
            task_meta=task_meta,
        )

        submission = pd.DataFrame({
            sample_df.columns[0]: test_df[id_col].values if id_col in test_df.columns else sample_df.iloc[:, 0].values,
            sample_df.columns[1]: pd.Series(formatted_preds, dtype="object"),
        })

        submission = self._sanitize_submission(submission, sample_df)

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
                *self.logs,
            ]
        )
        return submission, summary

    def _make_features(
        self,
        X: pd.DataFrame,
        raw_df: pd.DataFrame,
        description: str,
    ) -> pd.DataFrame:
        X = X.copy()

        # Generic numeric coercion
        for col in X.columns:
            if X[col].dtype == object:
                sample = X[col].dropna().astype(str).head(20)
                if len(sample) > 0 and sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.8:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

        # Spaceship Titanic-style features
        if "Cabin" in raw_df.columns and "Cabin" in X.columns:
            cabin = raw_df["Cabin"].astype(str).str.split("/", expand=True)
            if cabin.shape[1] >= 3:
                X["CabinDeck"] = cabin[0]
                X["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
                X["CabinSide"] = cabin[2]

        if "PassengerId" in raw_df.columns:
            pid = raw_df["PassengerId"].astype(str).str.split("_", expand=True)
            if pid.shape[1] >= 2:
                group_id = pid[0]
                X["GroupId"] = pd.to_numeric(group_id, errors="coerce")
                X["WithinGroupNo"] = pd.to_numeric(pid[1], errors="coerce")
                X["PassengerGroupSize"] = group_id.map(group_id.value_counts())

        if "Name" in raw_df.columns and "Name" in X.columns:
            name_series = raw_df["Name"].astype(str)
            X["NameLength"] = name_series.str.len()
            X["NameWordCount"] = name_series.str.split().str.len()

        spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in raw_df.columns]
        if spend_cols:
            for c in spend_cols:
                X[c] = pd.to_numeric(raw_df[c], errors="coerce")
            X["TotalSpend"] = X[spend_cols].sum(axis=1)
            X["NoSpend"] = (X["TotalSpend"].fillna(0) == 0).astype(int)
            luxury_cols = [c for c in ["Spa", "VRDeck"] if c in X.columns]
            if luxury_cols:
                X["LuxurySpend"] = X[luxury_cols].sum(axis=1)
            if "CryoSleep" in raw_df.columns:
                cryo = raw_df["CryoSleep"].astype(str).str.lower()
                X["CryoNoSpendMatch"] = (
                    (cryo.isin(["true", "1"])) & (X["NoSpend"] == 1)
                ).astype(int)

        if "Age" in raw_df.columns:
            X["Age"] = pd.to_numeric(raw_df["Age"], errors="coerce")
            X["AgeMissing"] = X["Age"].isna().astype(int)
            X["Age2"] = X["Age"] ** 2
            X["IsChild"] = (X["Age"] < 18).astype(float)
            X["IsSenior"] = (X["Age"] >= 60).astype(float)

        for col in ["HomePlanet", "Destination", "VIP", "CryoSleep"]:
            if col in raw_df.columns:
                X[col] = raw_df[col].astype(str)

        # Light generic text features
        object_cols = [c for c in X.columns if X[c].dtype == object]
        for col in object_cols[:6]:
            s = X[col].astype(str)
            X[f"{col}__len"] = s.str.len()
            X[f"{col}__missing_like"] = s.isin(["nan", "None", ""]).astype(int)

        # Remove raw high-cardinality identifiers except useful engineered features
        drop_cols = []
        for col in X.columns:
            if col.lower() in {"name", "cabin"}:
                drop_cols.append(col)
        if drop_cols:
            X = X.drop(columns=drop_cols, errors="ignore")

        return X

    def _align_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test = X_test.copy()
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = np.nan
        extra_cols = [c for c in X_test.columns if c not in X_train.columns]
        if extra_cols:
            X_test = X_test.drop(columns=extra_cols)
        return X_test[X_train.columns]

    def _prepare_target(self, y_raw: pd.Series, task_type: str) -> tuple[pd.Series, dict[str, Any]]:
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
                # stable binary mapping
                classes = sorted(pd.Series(y_raw.dropna().unique()).tolist(), key=lambda x: str(x))
                mapping = {classes[0]: 0, classes[1]: 1}
                y = y_raw.map(mapping).astype(int)
                meta["original_format"] = "binary_generic"
                meta["inverse_mapping"] = {0: classes[0], 1: classes[1]}
                meta["class_labels"] = [0, 1]
                meta["task_type"] = "binary_classification"
                return y, meta

            # multiclass
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

    def _build_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> list[tuple[str, Any]]:
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

        candidates: list[tuple[str, Any]] = []

        if task_type == "binary_classification":
            logreg = Pipeline(
                [
                    ("pre", pre_ohe),
                    ("model", LogisticRegression(max_iter=4000, C=1.5, solver="liblinear", random_state=42)),
                ]
            )
            rf = Pipeline(
                [
                    ("pre", pre_ohe),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=700,
                            max_depth=16,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            et = Pipeline(
                [
                    ("pre", pre_ohe),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=1000,
                            max_depth=None,
                            min_samples_leaf=1,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            hgb = Pipeline(
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
            )

            candidates.extend(
                [
                    ("logreg_ohe", logreg),
                    ("rf_ohe", rf),
                    ("extratrees_ohe", et),
                    ("hgb_ordinal", hgb),
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
                                (
                                    "model",
                                    RandomForestClassifier(
                                        n_estimators=700,
                                        max_depth=16,
                                        min_samples_leaf=2,
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
                                        n_estimators=1000,
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
                        "hgb_ordinal",
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
                                (
                                    "model",
                                    RandomForestRegressor(
                                        n_estimators=700,
                                        max_depth=16,
                                        min_samples_leaf=2,
                                        random_state=42,
                                        n_jobs=-1,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "extratrees_reg_ohe",
                        Pipeline(
                            [
                                ("pre", pre_ohe),
                                (
                                    "model",
                                    ExtraTreesRegressor(
                                        n_estimators=1000,
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
                        "hgb_reg_ordinal",
                        Pipeline(
                            [
                                ("pre", pre_ord),
                                (
                                    "model",
                                    HistGradientBoostingRegressor(
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
                ]
            )

        return candidates

    def _evaluate_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: list[tuple[str, Any]],
        task_type: str,
    ) -> CandidateResult:
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
                self.logs.append(f"{name}: {score:.6f}")
            except Exception as e:
                self.logs.append(f"{name}: FAILED ({e})")

        if not results:
            raise RuntimeError("All candidate models failed.")

        results.sort(key=lambda r: r.score, reverse=True)
        best = results[0]

        # Extra strong step for binary classification: build a soft-voting ensemble of top 3
        if task_type == "binary_classification":
            top = results[:3]
            ensemble_estimators = []
            for i, r in enumerate(top):
                model = r.model
                if hasattr(model, "predict_proba"):
                    ensemble_estimators.append((f"m{i}", model))

            if len(ensemble_estimators) >= 2:
                try:
                    ensemble = VotingClassifier(
                        estimators=ensemble_estimators,
                        voting="soft",
                        n_jobs=1,
                    )
                    cv = StratifiedKFold(n_splits=5 if len(y) >= 500 else 4, shuffle=True, random_state=42)
                    ens_score = float(np.mean(cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy", n_jobs=1)))
                    self.logs.append(f"soft_voting_top_models: {ens_score:.6f}")
                    if ens_score > best.score:
                        best = CandidateResult(
                            name="soft_voting_top_models",
                            score=ens_score,
                            model=ensemble,
                        )
                except Exception as e:
                    self.logs.append(f"soft_voting_top_models: FAILED ({e})")

        return best

    def _format_predictions(
        self,
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

    def _sanitize_submission(self, submission: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        fixed = sample_df.copy()
        pred_col = fixed.columns[1]
        fixed = fixed.astype({pred_col: "object"})

        original_sample_vals = (
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

        if set(original_sample_vals).issubset({"true", "false"}):
            fixed[pred_col] = (
                fixed[pred_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"1": "True", "0": "False", "true": "True", "false": "False"})
                .fillna("False")
            )
        elif set(original_sample_vals).issubset({"0", "1"}):
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

    def _build_fallback_submission(
        self,
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
    ) -> pd.DataFrame:
        fallback = sample_df.copy()
        id_col = sample_df.columns[0]

        if id_col in test_df.columns:
            n = min(len(fallback), len(test_df))
            fallback.iloc[:n, 0] = test_df.iloc[:n][id_col].values

        return self._sanitize_submission(fallback, sample_df)

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
