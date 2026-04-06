"""
Query the index by song name and return formatted recommendations.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def find_song_row(
    metadata: pd.DataFrame,
    song_name: str,
    artist_name: Optional[str] = None,
) -> Optional[int]:
    """
    Find the row index of a song by title (and optionally artist).
    Uses case-insensitive partial match on title.
    If multiple matches and artist_name given, prefer that artist.
    """
    title_col = "title" if "title" in metadata.columns else metadata.columns[0]
    meta = metadata.copy()
    meta["_title_lower"] = meta[title_col].astype(str).str.lower().str.strip()
    query_title = song_name.lower().strip()

    mask = meta["_title_lower"] == query_title
    matches = meta[mask]

    if matches.empty:
        return None
    if len(matches) == 1:
        return int(matches.index[0])
    if artist_name:
        artist_lower = artist_name.lower().strip()
        for idx in matches.index:
            if artist_lower in str(meta.loc[idx, "artist_name"]).lower().strip():
                return int(idx)
        return None
    return int(matches.index[0])


def recommend(
    nn_by_metric: dict,
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    song_name: str,
    top_k: int,
    artist_name: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Get top_k recommendations for a song by name.

    Returns:
        (recommendations_df, error_message). If song not found, recommendations_df is None
        and error_message is set. Otherwise error_message is None.
    """
    row_index = find_song_row(metadata, song_name, artist_name)
    if row_index is None:
        return None, "Song not found in the index. Please pick another song from the catalogue."

    query_vec = vectors[row_index : row_index + 1]
    k = min(top_k + 1, vectors.shape[0])

    
    recs_by_metric = {}

    for metric, nn in nn_by_metric.items():
        distances, indices = nn.kneighbors(query_vec, n_neighbors=k)

        neighbor_indices = indices[0].tolist()
        neighbor_indices = [i for i in neighbor_indices if i != row_index][:top_k]

        recs = metadata.iloc[neighbor_indices].reset_index(drop=True)
        recs_by_metric[metric] = recs
    
    return recs_by_metric, None


def format_recommendations(recs_by_metric: dict) -> str:
    lines = []

    for metric, recs in recs_by_metric.items():
        n = len(recs)

        lines.append(f"--- {metric.capitalize()} ---")
        lines.append(f"Here are your {n} recommendations:")
        lines.append("")

        for i, row in recs.iterrows():
            title = row.get("title", "Unknown")
            artist = row.get("artist_name", "Unknown")
            genre = row.get("genre", "")

            if pd.notna(genre) and str(genre).strip():
                lines.append(f"  {i + 1}. \"{title}\" — {artist} ({genre})")
            else:
                lines.append(f"  {i + 1}. \"{title}\" — {artist}")

        lines.append("") 

    return "\n".join(lines)
