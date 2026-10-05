"""Data loading and the single train/test split shared by every script."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Bankrupt?"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_csv(path):
    """Return (X, y, raw_df). Constant columns (e.g. 'Net Income Flag') are dropped from X."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")
    X = df.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.nunique(dropna=True) > 1]  # a constant column carries no information
    return X, y, df


def split_data(X, y):
    """Stratified 80/20 split. train.py, demo.py and eda.py all call this, so they agree."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
