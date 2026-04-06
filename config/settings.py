"""
Project settings: paths and feature list for the recommender.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

# Audio features used as the vector (must exist in MSD analysis.songs)
# See MSD schema: tempo, key, mode, loudness, duration, energy, danceability, etc.
FEATURE_COLUMNS = [
    "tempo",
    "key",
    "mode",
    "loudness",
    "duration",
    "energy",
    "danceability",
]

# Different distance metrics
METRICS = ["euclidean", "manhattan", "cosine"]
