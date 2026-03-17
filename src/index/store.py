"""
Save and load the index (scaler, vectors, metadata) to disk.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def save_index(
    scaler: StandardScaler,
    nn: NearestNeighbors,
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    index_dir: Path,
) -> None:
    """
    Save scaler, fitted NN model, vectors, and metadata to index_dir.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, index_dir / "scaler.joblib")
    joblib.dump(nn, index_dir / "nn.joblib")
    np.save(index_dir / "vectors.npy", vectors, allow_pickle=False)
    metadata.to_csv(index_dir / "metadata.csv", index=False)


def load_index(
    index_dir: Path,
) -> Tuple[StandardScaler, NearestNeighbors, np.ndarray, pd.DataFrame]:
    """
    Load scaler, NN model, vectors, and metadata from index_dir.
    """
    index_dir = Path(index_dir)
    scaler = joblib.load(index_dir / "scaler.joblib")
    nn = joblib.load(index_dir / "nn.joblib")
    vectors = np.load(index_dir / "vectors.npy", allow_pickle=False)
    metadata = pd.read_csv(index_dir / "metadata.csv")
    return scaler, nn, vectors, metadata
