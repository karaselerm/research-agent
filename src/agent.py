import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
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
        self.lesson_log: list[str] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = self._collect_payload(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Parsing competition payload...")
        )

        datasets = self._extract_named_csvs(payload)
        train_df = datasets.get("train")
        test_df = datasets.get("test")
        sample_submission_df = datasets.get("sample_submission")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=payload[:30000]))],
            name="debug_input_preview.txt",
        )

        if sample_submission_df is None:
            raise ValueError("Could not find sample_submission.csv in the incoming payload.")

        if train_df is None or test_df is None:
            fallback = self._fill_sample_submission(sample_submission_df)
            await self._add_csv_artifact(updater, "submission.csv", fallback)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Used fallback sample_submission because train/test were not found."))],
                name="debug_strategy.txt",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing schema and preparing candidate models...")
        )

        target_col = self._detect_target_column(train_df, sample_submission_df)
        id_col = self._detect_id_column(test_df, sample_submission_df)
        pred_col = self._detect_submission_prediction_column(sample_submission_df, target_col)

        if target_col is None:
            fallback = self._fill_sample_submission(sample_submission_df)
            await self._add_csv_artifact(updater, "submission.csv", fallback)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Target column not detected; used fallback sample_submission."))],
                name="debug_strategy.txt",
            )
            return

        train_df = train_df.copy()
        test_df = test_df.copy()

        y_raw = train_df[target_col].copy()
        X_train = train_df.drop(columns=[target_col], errors="ignore")
        X_test = test_df.copy()

        if id_col and id_col in X_train.columns:
            X_train = X_train.drop(columns=[id_col], errors="ignore")
        if id_col and id_col in X_test.columns:
            X_test_features = X_test.drop(columns=[id_col], errors="ignore")
        else:
            X_test_features = X_test.copy()

        X_train = self._feature_engineer(X_train, train_df)
        X_test_features = self._feature_engineer(X_test_features, test_df)
        X_test_features = self._align_test_to_train(X_train, X_test_features)

        y, target_format = self._coerce_target(y_raw)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Evaluating candidate pipelines with cross-validation...")
        )

        best = self._select_best_candidate(X_train, y)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Best candidate: {best.name} (CV={best.score:.4f}). Training final model...")
        )

        best.model.fit(X_train, y)
        preds = best.model.predict(X_test_features)

        submission = self._build_submission(
            test_df=test_df,
            sample_submission_df=sample_submission_df,
            id_col=id_col,
            pred_col=pred_col,
            preds=preds,
            target_format=target_format,
        )

        submission = self._sanitize_submission_against_sample(
            submission=submission,
            sample_submission_df=sample_submission_df,
        )

        debug_report = [
            f"target_col={target_col}",
            f"id_col={id_col}",
            f"pred_col={pred_col}",
            f"target_format={target_format}",
            f"train_shape={train_df.shape}",
            f"test_shape={test_df.shape}",
            f"submission_shape={submission.shape}",
            f"best_model={best.name}",
            f"best_cv={best.score:.6f}",
            "",
            "candidate_logs:",
            *self.lesson_log,
        ]

        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n".join(debug_report)))],
            name="debug_report.txt",
        )

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=submission.head(20).to_csv(index=False)))],
            name="debug_submission_preview.csv",
        )

        await self._add_csv_artifact(updater, "submission.csv", submission)

    async def _add_csv_artifact(self, updater: TaskUpdater, filename: str, df: pd.DataFrame) -> None:
        csv_text = df.to_csv(index=False)
        file_bytes_b64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
        file_part = FilePart(
            file=FileWithBytes(
                name=filename,
                mimeType="text/csv",
                bytes=file_bytes_b64,
            )
        )
        await updater.add_artifact(
            parts=[Part(root=file_part)],
            name=filename,
        )

    def _collect_payload(self, message: Message) -> str:
        chunks: list[str] = []
        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                if root.text:
                    chunks.append(root.text)
            elif isinstance(root, DataPart):
                try:
                    chunks.append(json.dumps(root.data, ensure_ascii=False, indent=2))
                except Exception:
                    chunks.append(str(root.data))
            else:
                chunks.append(str(root))
        return "\n\n".join(chunks)

    def _extract_named_csvs(self, text: str) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for name in ["train", "test", "sample_submission"]:
            df = self._find_csv_after_filename(text, name)
            if df is not None:
                out[name] = df

        if "sample_submission" not in out:
            sample_df = self._find_generic_sample_submission(text)
            if sample_df is not None:
                out["sample_submission"] = sample_df

        return out

    def _find_csv_after_filename(self, text: str, base_name: str) -> pd.DataFrame | None:
        patterns = [
            rf"{re.escape(base_name)}\.csv.*?```(?:csv)?\n(.*?)```",
            rf"{re.escape(base_name)}\.csv.*?\n((?:[^\n]*,[^\n]*\n)+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            for block in matches:
                df = self._safe_read_csv(block)
                if df is not None and not df.empty:
                    return df
        return None

    def _find_generic_sample_submission(self, text: str) -> pd.DataFrame | None:
        patterns = [
            r"```(?:csv)?\n(PassengerId\s*,\s*Transported.*?)(?:```)",
            r"```(?:csv)?\n(id\s*,\s*target.*?)(?:```)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            for block in matches:
                df = self._safe_read_csv(block)
                if df is not None and not df.empty:
                    return df
        return None

    def _safe_read_csv(self, raw_csv: str) -> pd.DataFrame | None:
        cleaned = self._trim_csv_like_text(raw_csv)
        if not cleaned:
            return None
        try:
            return pd.read_csv(io.StringIO(cleaned))
        except Exception:
            return None

    def _trim_csv_like_text(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        csv_lines = [line for line in lines if "," in line]
        return "\n".join(csv_lines)

    def _detect_target_column(self, train_df: pd.DataFrame, sample_submission_df: pd.DataFrame | None) -> str | None:
        preferred = ["Transported", "target", "label", "Survived", "Outcome", "Response"]
        for col in preferred:
            if col in train_df.columns:
                return col

        if sample_submission_df is not None and len(sample_submission_df.columns) >= 2:
            cand = sample_submission_df.columns[1]
            if cand in train_df.columns:
                return cand

        for col in reversed(train_df.columns):
            if "id" in col.lower():
                continue
            nunique = train_df[col].nunique(dropna=True)
            if 2 <= nunique <= 20:
                return col

        return None

    def _detect_id_column(self, test_df: pd.DataFrame, sample_submission_df: pd.DataFrame) -> str:
        preferred = ["PassengerId", "id", "Id", "ID", "row_id"]
        for col in preferred:
            if col in test_df.columns:
                return col

        first = sample_submission_df.columns[0]
        if first in test_df.columns:
            return first

        return first

    def _detect_submission_prediction_column(self, sample_submission_df: pd.DataFrame, target_col: str) -> str:
        if len(sample_submission_df.columns) >= 2:
            return sample_submission_df.columns[1]
        return target_col

    def _feature_engineer(self, X: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "Cabin" in raw_df.columns and "Cabin" in X.columns:
            cabin_parts = raw_df["Cabin"].astype(str).str.split("/", expand=True)
            if cabin_parts.shape[1] >= 3:
                X["CabinDeck"] = cabin_parts[0]
                X["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
                X["CabinSide"] = cabin_parts[2]

        if "Name" in raw_df.columns and "Name" in X.columns:
            X["NameLength"] = raw_df["Name"].astype(str).str.len()

        if "PassengerId" in raw_df.columns:
            pid_parts = raw_df["PassengerId"].astype(str).str.split("_", expand=True)
            if pid_parts.shape[1] >= 2:
                X["GroupId"] = pd.to_numeric(pid_parts[0], errors="coerce")
                X["WithinGroupNo"] = pd.to_numeric(pid_parts[1], errors="coerce")
                X["PassengerGroupSize"] = raw_df.groupby(pid_parts[0])[pid_parts[0]].transform("count").values

        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        existing = [c for c in spend_cols if c in raw_df.columns]
        if existing:
            for c in existing:
                X[c] = pd.to_numeric(raw_df[c], errors="coerce")
            X["TotalSpend"] = X[existing].sum(axis=1)
            X["NoSpend"] = (X["TotalSpend"].fillna(0) == 0).astype(int)
            luxury = [c for c in ["Spa", "VRDeck"] if c in X.columns]
            if luxury:
                X["LuxurySpend"] = X[luxury].sum(axis=1)

        if "Age" in raw_df.columns:
            X["Age"] = pd.to_numeric(raw_df["Age"], errors="coerce")
            X["AgeMissing"] = X["Age"].isna().astype(int)
            X["AgeBinTeen"] = ((X["Age"] >= 12) & (X["Age"] < 20)).astype(int)
            X["AgeBinSenior"] = (X["Age"] >= 60).astype(int)

        for col in ["CryoSleep", "VIP", "HomePlanet", "Destination"]:
            if col in raw_df.columns:
                X[col] = raw_df[col].astype(str)

        return X

    def _align_test_to_train(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test = X_test.copy()
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = np.nan
        extra_cols = [c for c in X_test.columns if c not in X_train.columns]
        if extra_cols:
            X_test = X_test.drop(columns=extra_cols)
        return X_test[X_train.columns]

    def _coerce_target(self, y: pd.Series) -> tuple[pd.Series, str]:
        if y.dtype == bool:
            return y.astype(int), "bool"

        lowered = y.astype(str).str.lower().str.strip()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            return lowered.map({"true": 1, "false": 0}).astype(int), "truefalse"

        if set(y.dropna().unique()).issubset({0, 1}):
            return y.astype(int), "int01"

        return y, "raw"

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]

        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical,
                ),
            ]
        )

    def _candidate_models(self, X: pd.DataFrame) -> list[tuple[str, Pipeline]]:
        pre = self._build_preprocessor(X)

        return [
            (
                "logreg",
                Pipeline([
                    ("pre", pre),
                    ("model", LogisticRegression(max_iter=3000, C=1.0, solver="liblinear", random_state=42)),
                ]),
            ),
            (
                "rf",
                Pipeline([
                    ("pre", pre),
                    ("model", RandomForestClassifier(
                        n_estimators=500,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    )),
                ]),
            ),
            (
                "extratrees",
                Pipeline([
                    ("pre", pre),
                    ("model", ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    )),
                ]),
            ),
            (
                "hgb",
                Pipeline([
                    ("pre", pre),
                    ("model", HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.05,
                        max_iter=300,
                        random_state=42,
                    )),
                ]),
            ),
        ]

    def _select_best_candidate(self, X: pd.DataFrame, y: pd.Series) -> CandidateResult:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        results: list[CandidateResult] = []

        for name, model in self._candidate_models(X):
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
                score = float(np.mean(scores))
                results.append(CandidateResult(name=name, score=score, model=model))
                self.lesson_log.append(f"{name}: CV={score:.4f}")
            except Exception as e:
                self.lesson_log.append(f"{name} failed: {e}")

        if not results:
            raise RuntimeError("All candidate models failed during cross-validation.")

        results.sort(key=lambda r: r.score, reverse=True)
        return results[0]

    def _build_submission(
        self,
        test_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        id_col: str,
        pred_col: str,
        preds: np.ndarray,
        target_format: str,
    ) -> pd.DataFrame:
        submission = pd.DataFrame()

        sample_id_col = sample_submission_df.columns[0]
        sample_pred_col = sample_submission_df.columns[1]

        if id_col in test_df.columns:
            submission[sample_id_col] = test_df[id_col].values
        else:
            submission[sample_id_col] = sample_submission_df.iloc[:, 0].values

        submission[sample_pred_col] = self._format_predictions(
            preds=preds,
            target_format=target_format,
            sample_submission_df=sample_submission_df,
            pred_col=sample_pred_col,
        )

        return submission

    def _format_predictions(
        self,
        preds: np.ndarray,
        target_format: str,
        sample_submission_df: pd.DataFrame,
        pred_col: str,
    ) -> pd.Series:
        preds = pd.Series(preds)

        sample_vals = (
            sample_submission_df[pred_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )

        if target_format in {"bool", "truefalse"} or set(sample_vals).issubset({"true", "false"}):
            return preds.astype(int).map({1: "True", 0: "False"})

        if target_format == "int01" or set(sample_vals).issubset({"0", "1"}):
            return preds.astype(int)

        return preds

    def _sanitize_submission_against_sample(
        self,
        submission: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
    ) -> pd.DataFrame:
        sample_cols = list(sample_submission_df.columns)
        submission = submission.copy()

        if len(sample_cols) < 2:
            raise ValueError("sample_submission.csv has invalid format.")

        if sample_cols[0] not in submission.columns or sample_cols[1] not in submission.columns:
            fixed = sample_submission_df.copy()
            if submission.shape[1] >= 2:
                fixed.iloc[:, 0] = submission.iloc[:, 0].values[: len(fixed)]
                fixed.iloc[:, 1] = submission.iloc[:, 1].values[: len(fixed)]
            submission = fixed

        submission = submission[sample_cols[:2]]

        if len(submission) != len(sample_submission_df):
            fixed = sample_submission_df.copy()
            n = min(len(submission), len(fixed))
            fixed.iloc[:n, 0] = submission.iloc[:n, 0].values
            fixed.iloc[:n, 1] = submission.iloc[:n, 1].values
            submission = fixed

        pred_col = sample_cols[1]
        sample_vals = (
            sample_submission_df[pred_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )

        if set(sample_vals).issubset({"true", "false"}):
            mapped = (
                submission[pred_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "1": "True",
                    "0": "False",
                    "true": "True",
                    "false": "False",
                })
            )
            submission[pred_col] = mapped.fillna("False")

        elif set(sample_vals).issubset({"0", "1"}):
            mapped = (
                submission[pred_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "1": 1,
                    "0": 0,
                    "true": 1,
                    "false": 0,
                })
            )
            submission[pred_col] = mapped.fillna(0).astype(int)

        return submission

    def _fill_sample_submission(self, sample_submission_df: pd.DataFrame) -> pd.DataFrame:
        df = sample_submission_df.copy()

        if len(df.columns) < 2:
            raise ValueError("Sample submission has fewer than 2 columns.")

        pred_col = df.columns[1]
        values = df[pred_col].dropna().astype(str).str.strip().str.lower().unique().tolist()

        if set(values).issubset({"true", "false"}):
            df[pred_col] = "False"
        elif set(values).issubset({"0", "1"}):
            df[pred_col] = 0
        else:
            df[pred_col] = df[pred_col].fillna(0)

        return df