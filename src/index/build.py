"""
Build the recommendation index: scale features and fit nearest-neighbor search.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from config.settings import METRICS


def build_index(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[StandardScaler, dict, np.ndarray, pd.DataFrame]:
    
    """
    Build the index from a DataFrame of songs.

    Args:
        df: DataFrame with feature columns and metadata (track_id, title, artist_name, genre).
        feature_columns: Column names to use as the vector.

    Returns:
        (scaler, nn_model, vectors_scaled, metadata_df) where metadata_df has track_id, title,
        artist_name, genre and the same row order as vectors_scaled.
    """
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError(f"None of {feature_columns} found in DataFrame. Columns: {list(df.columns)}")

    X = df[available].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn_by_metric = {}

    for m in METRICS: 
        nn = NearestNeighbors(metric=m, algorithm="auto")
        nn.fit(X_scaled)
        nn_by_metric[m] = nn

    meta_cols = ["track_id", "title", "artist_name", "genre"]
    metadata = df[[c for c in meta_cols if c in df.columns]].copy()
    if "genre" not in metadata.columns:
        metadata["genre"] = ""

    return scaler, nn_by_metric, X_scaled, metadata