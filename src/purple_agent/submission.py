import pandas as pd


def sanitize_submission(
    submission: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
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


def build_fallback_submission(
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    fallback = sample_df.copy()
    id_col = sample_df.columns[0]

    if id_col in test_df.columns:
        n = min(len(fallback), len(test_df))
        fallback.iloc[:n, 0] = test_df.iloc[:n][id_col].values

    return sanitize_submission(fallback, sample_df)


def validate_submission(submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
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
