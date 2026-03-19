import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# CONFIG
# ==============================

BASE_URL = "https://huggingface.co/datasets/cosita2000/index/resolve/main/"

FILES = [
    "metadata.csv",
    "vectors.npy",
    "scaler.joblib",
    "nn_cosine.joblib",
    "nn_euclidean.joblib",
    "nn_manhattan.joblib"
]

DATA_DIR = "index_data"

# ==============================
# DOWNLOAD + LOAD
# ==============================

@st.cache_resource
def load_assets():
    os.makedirs(DATA_DIR, exist_ok=True)

    for file in FILES:
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {file}..."):
                url = BASE_URL + file
                r = requests.get(url)
                if r.status_code != 200:
                    raise Exception(f"Failed to download {file}")
                with open(path, "wb") as f:
                    f.write(r.content)

    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # Rename columns to match your metadata standard
    if "song" in metadata.columns:
        metadata = metadata.rename(columns={"song": "title"})
    if "artist" in metadata.columns:
        metadata = metadata.rename(columns={"artist": "artist_name"})

    vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.joblib"))

    nn_models = {
        "Cosine": joblib.load(os.path.join(DATA_DIR, "nn_cosine.joblib")),
        "Euclidean": joblib.load(os.path.join(DATA_DIR, "nn_euclidean.joblib")),
        "Manhattan": joblib.load(os.path.join(DATA_DIR, "nn_manhattan.joblib")),
    }

    return metadata, vectors, scaler, nn_models


# ==============================
# RECOMMENDER
# ==============================

def recommend(song_name, metadata, vectors, scaler, nn_model, artist_name=None, top_k=5):
    """
    Finds similar songs based on a song name and optional artist name.
    """
    matches = metadata[metadata["title"].str.lower() == song_name.lower()]

    if artist_name:
        matches = matches[matches["artist_name"].str.lower() == artist_name.lower()]

    if len(matches) == 0:
        return ["Song not found. Try another name."]

    idx = matches.index[0]
    query_vec = vectors[idx].reshape(1, -1)
    query_vec = scaler.transform(query_vec)
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_k + 1)

    results = []
    for i in indices[0]:
        if i != idx:
            row = metadata.iloc[i]
            results.append(f"{row['title']} - {row.get('artist_name', 'Unknown')}")
    return results[:top_k]


# ==============================
# UI
# ==============================

st.set_page_config(page_title="Song Recommender", layout="centered")
st.title("🎵 Song Recommender")
st.write("Type a song name and optionally an artist, then click Search.")

# Load data
metadata, vectors, scaler, nn_models = load_assets()

# Input fields
query_song = st.text_input("Enter song name")
query_artist = st.text_input("Enter artist name (optional)")

# Search button
if st.button("Search"):
    if not query_song:
        st.warning("Please enter a song name.")
    else:
        with st.spinner("Finding recommendations..."):
            for metric_name, model in nn_models.items():
                results = recommend(query_song, metadata, vectors, scaler, model, artist_name=query_artist)
                st.subheader(f"Recommendations ({metric_name})")
                for i, r in enumerate(results, 1):
                    st.write(f"{i}. {r}")