import csv
import io
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = self._collect_payload(message)

        print("=== INPUT PAYLOAD START ===")
        print(payload[:50000])
        print("=== INPUT PAYLOAD END ===")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Parsing competition payload and searching for datasets...")
        )

        datasets = self._extract_named_csvs(payload)

        train_df = datasets.get("train")
        test_df = datasets.get("test")
        sample_submission_df = datasets.get("sample_submission")

        if train_df is not None and test_df is not None:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Train/test data found. Building a model...")
            )

            submission_df = self._build_ml_submission(
                train_df=train_df,
                test_df=test_df,
                sample_submission_df=sample_submission_df,
            )
        elif sample_submission_df is not None:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("No train/test pair found. Falling back to sample submission baseline...")
            )
            submission_df = self._fill_sample_submission(sample_submission_df)
        else:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=payload[:50000]))],
                name="debug_input.txt",
            )
            raise ValueError("Could not find usable train/test/sample_submission CSV content in incoming payload.")

        csv_text = submission_df.to_csv(index=False)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=csv_text))],
            name="submission.csv",
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"submission.csv prepared with {len(submission_df)} rows."
            ),
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
        """
        Tries to find blocks corresponding to train/test/sample_submission.
        Works with patterns like:
        - train.csv ... csv content
        - sample_submission.csv ... csv content
        - fenced code blocks
        """
        results: dict[str, pd.DataFrame] = {}

        for name in ["train", "test", "sample_submission"]:
            df = self._find_csv_after_filename(text, name)
            if df is not None:
                results[name] = df

        # Extra fallback: if sample submission not found explicitly, search by common header
        if "sample_submission" not in results:
            sample_df = self._find_generic_sample_submission(text)
            if sample_df is not None:
                results["sample_submission"] = sample_df

        return results

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
        generic_patterns = [
            r"```(?:csv)?\n(PassengerId\s*,\s*Transported.*?)(?:```)",
            r"```(?:csv)?\n(id\s*,\s*target.*?)(?:```)",
        ]

        for pattern in generic_patterns:
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

    def _build_ml_submission(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        target_col = self._detect_target_column(train_df, sample_submission_df)
        id_col = self._detect_id_column(train_df, test_df, sample_submission_df, target_col)

        if target_col is None:
            if sample_submission_df is not None:
                return self._fill_sample_submission(sample_submission_df)
            raise ValueError("Could not detect target column.")

        y = train_df[target_col].copy()
        X = train_df.drop(columns=[target_col], errors="ignore").copy()
        X_test = test_df.copy()

        # Align columns
        if id_col and id_col in X.columns:
            X = X.drop(columns=[id_col], errors="ignore")
        if id_col and id_col in X_test.columns:
            X_test_features = X_test.drop(columns=[id_col], errors="ignore")
        else:
            X_test_features = X_test.copy()

        X = self._feature_engineer(X)
        X_test_features = self._feature_engineer(X_test_features)

        # Re-align after feature engineering
        missing_cols = [c for c in X.columns if c not in X_test_features.columns]
        for c in missing_cols:
            X_test_features[c] = np.nan

        extra_cols = [c for c in X_test_features.columns if c not in X.columns]
        if extra_cols:
            X_test_features = X_test_features.drop(columns=extra_cols)

        X_test_features = X_test_features[X.columns]

        model = self._build_model(X, y)
        model.fit(X, y)
        preds = model.predict(X_test_features)

        pred_col = self._detect_submission_prediction_column(sample_submission_df, target_col)
        out = pd.DataFrame()

        if id_col and id_col in test_df.columns:
            out[id_col] = test_df[id_col]
        elif sample_submission_df is not None and len(sample_submission_df.columns) >= 1:
            out[sample_submission_df.columns[0]] = sample_submission_df.iloc[:, 0]
        else:
            out["id"] = np.arange(len(test_df))

        out[pred_col] = preds

        # Convert boolean-ish target to Kaggle-friendly values
        out[pred_col] = self._normalize_prediction_series(out[pred_col], sample_submission_df, pred_col)

        return out

    def _detect_target_column(
        self,
        train_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame | None,
    ) -> str | None:
        preferred = [
            "Transported", "target", "label", "Survived", "Response", "Outcome"
        ]

        for col in preferred:
            if col in train_df.columns:
                return col

        if sample_submission_df is not None and len(sample_submission_df.columns) >= 2:
            candidate = sample_submission_df.columns[1]
            if candidate in train_df.columns:
                return candidate

        # fallback: last column with low-cardinality or binary-ish values
        for col in reversed(train_df.columns):
            nunique = train_df[col].nunique(dropna=True)
            if nunique <= 20:
                return col

        return None

    def _detect_id_column(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame | None,
        target_col: str | None,
    ) -> str | None:
        preferred = ["PassengerId", "id", "Id", "ID", "row_id"]

        for col in preferred:
            if col in test_df.columns:
                return col

        if sample_submission_df is not None and len(sample_submission_df.columns) >= 1:
            first = sample_submission_df.columns[0]
            if first in test_df.columns:
                return first

        for col in test_df.columns:
            if col != target_col and col not in train_df.select_dtypes(include=[np.number]).columns:
                return col

        return None

    def _detect_submission_prediction_column(
        self,
        sample_submission_df: pd.DataFrame | None,
        target_col: str,
    ) -> str:
        if sample_submission_df is not None and len(sample_submission_df.columns) >= 2:
            return sample_submission_df.columns[1]
        return target_col

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Spaceship Titanic useful handcrafted features
        if "Cabin" in df.columns:
            cabin_parts = df["Cabin"].astype(str).str.split("/", expand=True)
            if cabin_parts.shape[1] >= 3:
                df["CabinDeck"] = cabin_parts[0]
                df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
                df["CabinSide"] = cabin_parts[2]

        if "Name" in df.columns:
            df["NameLength"] = df["Name"].astype(str).str.len()

        if "PassengerId" in df.columns:
            pid_parts = df["PassengerId"].astype(str).str.split("_", expand=True)
            if pid_parts.shape[1] >= 2:
                df["GroupId"] = pd.to_numeric(pid_parts[0], errors="coerce")
                df["GroupSizeHint"] = df.groupby(pid_parts[0])[pid_parts[0]].transform("count")

        if "CryoSleep" in df.columns:
            df["CryoSleep"] = df["CryoSleep"].astype(str)

        if "VIP" in df.columns:
            df["VIP"] = df["VIP"].astype(str)

        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        existing_spend = [c for c in spend_cols if c in df.columns]
        if existing_spend:
            for c in existing_spend:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["TotalSpend"] = df[existing_spend].sum(axis=1)
            df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)

        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
            df["AgeMissing"] = df["Age"].isna().astype(int)

        return df

    def _build_model(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        y_clean = self._coerce_target(y)

        numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

        lr = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="liblinear",
            random_state=42,
        )

        ensemble = VotingClassifier(
            estimators=[
                ("rf", rf),
                ("et", et),
                ("lr", lr),
            ],
            voting="soft",
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", ensemble),
            ]
        )

        # return pipeline expecting cleaned y externally
        pipe._y_clean = y_clean  # only for debugging if needed
        return WrappedPipeline(pipe, y_clean)

    def _coerce_target(self, y: pd.Series) -> pd.Series:
        if y.dtype == bool:
            return y.astype(int)

        lowered = y.astype(str).str.lower().str.strip()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            return lowered.map({"true": 1, "false": 0}).astype(int)

        if set(y.dropna().unique()).issubset({0, 1}):
            return y.astype(int)

        return y

    def _normalize_prediction_series(
        self,
        series: pd.Series,
        sample_submission_df: pd.DataFrame | None,
        pred_col: str,
    ) -> pd.Series:
        if sample_submission_df is None or pred_col not in sample_submission_df.columns:
            return series

        sample_values = sample_submission_df[pred_col].dropna().astype(str).str.strip().str.lower().unique().tolist()

        if set(sample_values).issubset({"true", "false"}):
            return pd.Series(series).astype(int).map({1: "True", 0: "False"})

        if set(sample_values).issubset({"0", "1"}):
            return pd.Series(series).astype(int)

        return series

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


class WrappedPipeline:
    """
    Tiny adapter so we can coerce target before fit while keeping sklearn-style API.
    """

    def __init__(self, pipeline: Pipeline, y_clean: pd.Series):
        self.pipeline = pipeline
        self.y_clean = y_clean

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, self.y_clean)
        return self

    def predict(self, X: pd.DataFrame):
        preds = self.pipeline.predict(X)
        return preds