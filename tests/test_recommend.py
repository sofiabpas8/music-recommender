"""Basic tests for recommend module (no MSD data required)."""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommend import find_song_row, format_recommendations


def test_find_song_row():
    metadata = pd.DataFrame({
        "title": ["Song A", "Song B", "Song A"],
        "artist_name": ["Artist 1", "Artist 2", "Artist 3"],
    })
    assert find_song_row(metadata, "Song A") is not None
    assert find_song_row(metadata, "Song B") is not None
    assert find_song_row(metadata, "Song A", "Artist 3") == 2
    assert find_song_row(metadata, "Not There") is None


def test_format_recommendations():
    recs = pd.DataFrame({
        "title": ["R1", "R2"],
        "artist_name": ["A1", "A2"],
        "genre": ["Pop", ""],
    })
    out = format_recommendations(recs)
    assert "Here are your 2 recommendations" in out
    assert "R1" in out and "A1" in out
    assert "Pop" in out
