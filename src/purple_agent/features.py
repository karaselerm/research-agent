import numpy as np
import pandas as pd


def make_features(
    X: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == object:
            sample = X[col].dropna().astype(str).head(20)
            if len(sample) > 0 and sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.8:
                X[col] = pd.to_numeric(X[col], errors="coerce")

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

    spend_cols = [
        c
        for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        if c in raw_df.columns
    ]
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

    object_cols = [c for c in X.columns if X[c].dtype == object]
    for col in object_cols[:6]:
        s = X[col].astype(str)
        X[f"{col}__len"] = s.str.len()
        X[f"{col}__missing_like"] = s.isin(["nan", "None", ""]).astype(int)

    drop_cols = [col for col in X.columns if col.lower() in {"name", "cabin"}]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

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
