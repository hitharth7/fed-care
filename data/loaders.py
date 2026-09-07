"""Load and cache the CDC Diabetes Health Indicators dataset (UCI id=891)."""
import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
RAW_PATH = RAW_DIR / "diabetes.csv"

UCI_DATASET_ID = 891
TARGET_COL = "Diabetes_binary"


def _ensure_ssl_certs() -> None:
    # python.org macOS builds don't wire a CA bundle into ssl by default, which
    # breaks the plain urllib fetch ucimlrepo uses. Point it at certifi's bundle.
    if "SSL_CERT_FILE" not in os.environ:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()


def load_diabetes_data(force_download: bool = False) -> pd.DataFrame:
    """Return the CDC Diabetes Health Indicators dataset as one DataFrame
    (21 feature columns + `Diabetes_binary` target), fetching from UCI on
    first call and caching to data/raw/diabetes.csv afterwards.
    """
    if RAW_PATH.exists() and not force_download:
        return pd.read_csv(RAW_PATH)

    _ensure_ssl_certs()
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def load_features_targets(force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with y as the binary Diabetes_binary target."""
    df = load_diabetes_data(force_download=force_download)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y
