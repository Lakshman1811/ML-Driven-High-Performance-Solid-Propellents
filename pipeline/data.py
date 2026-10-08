from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_CANDIDATES, FEATURE_COLUMNS


def find_dataset() -> Path:
    for path in DATA_CANDIDATES:
        if path.is_file():
            return path
    tried = "\n".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Supplementary Data 1.csv not found. Place it at data/Supplementary Data 1.csv.\nTried:\n"
        + tried
    )


def load_raw() -> pd.DataFrame:
    path = find_dataset()
    df = pd.read_csv(path)
    split_col = df.columns[0]
    df = df.rename(columns={split_col: "split"})
    df = df.loc[~((df["isp"] == 0) | (df["cstar"] == 0))].copy()
    return df


def xy_from_frame(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = {
        "c_t": df["c_t"].to_numpy(np.float32),
        "isp": df["isp"].to_numpy(np.float32),
        "cstar": df["cstar"].to_numpy(np.float32),
    }
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler
