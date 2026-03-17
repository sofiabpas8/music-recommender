"""
Build the recommendation index: scale features and fit nearest-neighbor search.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_index(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_neighbors: int = 6,
) -> Tuple[StandardScaler, NearestNeighbors, np.ndarray, pd.DataFrame]:
    """
    Build the index from a DataFrame of songs.

    Args:
        df: DataFrame with feature columns and metadata (track_id, title, artist_name, genre).
        feature_columns: Column names to use as the vector.
        n_neighbors: K+1 for search (so we can exclude query and return K).

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

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(X_scaled)

    meta_cols = ["track_id", "title", "artist_name", "genre"]
    metadata = df[[c for c in meta_cols if c in df.columns]].copy()
    if "genre" not in metadata.columns:
        metadata["genre"] = ""

    return scaler, nn, X_scaled, metadata


def get_neighbors(
    scaler: StandardScaler,
    nn: NearestNeighbors,
    metadata: pd.DataFrame,
    query_row_index: int,
    vectors: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Get top_k nearest neighbours for the song at query_row_index, excluding the query itself.

    Args:
        scaler: Fitted StandardScaler (not used if vectors already scaled).
        nn: Fitted NearestNeighbors.
        metadata: DataFrame with same row order as vectors.
        query_row_index: Row index of the query song.
        vectors: Scaled feature matrix (n_songs, n_features).
        top_k: Number of recommendations to return.

    Returns:
        DataFrame of recommended songs (title, artist_name, genre) with top_k rows.
    """
    query_vec = vectors[query_row_index : query_row_index + 1]
    k = min(top_k + 1, vectors.shape[0])
    distances, indices = nn.kneighbors(query_vec, n_neighbors=k)

    # Exclude the query (first neighbour is self)
    neighbor_indices = indices[0].tolist()
    if query_row_index in neighbor_indices:
        neighbor_indices = [i for i in neighbor_indices if i != query_row_index]
    neighbor_indices = neighbor_indices[:top_k]

    return metadata.iloc[neighbor_indices].reset_index(drop=True)
