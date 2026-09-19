"""Shared data-loading and feature logic, so training and prediction can never drift apart."""
import pandas as pd

TARGET = "Stay/Left"

# Columns from the raw CSV that are identifiers / personal data -> never used as features
DROP_COLS = ["table id", "name", "phone number"]

CATEGORICAL = [
    "Location",
    "Emp. Group",
    "Function",
    "Gender",
    "Marital Status",
    "Hiring Source",
    "Promoted/Non Promoted",
    "Job Role Match",
]
NUMERIC = ["Experience_years", "Age_years", "Tenure_years"]
FEATURES = CATEGORICAL + NUMERIC


def yymm_to_years(x):
    """The dataset stores durations as YY.MM (e.g. 6.08 = 6 years 8 months), NOT decimal years.
    Convert to real decimal years so 'x.08' and 'x.10' are ordered and spaced correctly."""
    if pd.isna(x):
        return x
    years = int(x)
    months = round((x - years) * 100)
    return years + months / 12


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # e.g. "Gender " has a trailing space
    return df


def prepare(df):
    """Raw dataframe -> (X, y). y is 1 if the employee Left, else 0."""
    df = df.copy()
    df["Experience_years"] = df["Experience (YY.MM)"].apply(yymm_to_years)
    df["Age_years"] = df["Age in YY."].apply(yymm_to_years)
    df["Tenure_years"] = df["Tenure"].apply(yymm_to_years)
    X = df[FEATURES]
    y = (df[TARGET] == "Left").astype(int) if TARGET in df else None
    return X, y
