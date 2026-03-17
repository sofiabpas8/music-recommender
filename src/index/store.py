"""
Save and load the index (scaler, vectors, metadata) to disk.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def save_index(
    scaler: StandardScaler,
    #nn: nn_by_metric,
    nn_by_metric: Dict[str, NearestNeighbors],
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
    #joblib.dump(nn, index_dir / "nn.joblib")
    for metric, nn in nn_by_metric.items():
        joblib.dump(nn, index_dir / f"nn_{metric}.joblib")
    with open(index_dir / "metrics.txt", "w") as f:
        for metric in nn_by_metric.keys():
            f.write(f"{metric}\n")
    np.save(index_dir / "vectors.npy", vectors, allow_pickle=False)
    metadata.to_csv(index_dir / "metadata.csv", index=False)


def load_index(
    index_dir: Path,
    #) -> Tuple[StandardScaler, NearestNeighbors, np.ndarray, pd.DataFrame]:
) -> Tuple[StandardScaler, Dict[str, NearestNeighbors], np.ndarray, pd.DataFrame]:
    """
    Load scaler, NN model, vectors, and metadata from index_dir.
    """
    index_dir = Path(index_dir)
    scaler = joblib.load(index_dir / "scaler.joblib")
    nn_by_metric: Dict[str, NearestNeighbors] = {}
    metrics_file = index_dir / "metrics.txt"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = [line.strip() for line in f if line.strip()]

        for metric in metrics:
            nn_by_metric[metric] = joblib.load(index_dir / f"nn_{metric}.joblib")
    else:
        for path in index_dir.glob("nn_*.joblib"):
            metric = path.stem.replace("nn_", "")
            nn_by_metric[metric] = joblib.load(path)
    #nn = joblib.load(index_dir / "nn.joblib")
    vectors = np.load(index_dir / "vectors.npy", allow_pickle=False)
    metadata = pd.read_csv(index_dir / "metadata.csv")
    return scaler, nn_by_metric, vectors, metadata
