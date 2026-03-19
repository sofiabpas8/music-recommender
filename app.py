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

    # Load everything
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # Rename columns if needed
    if "song" in metadata.columns:
        metadata = metadata.rename(columns={"song": "title"})
    if "artist" in metadata.columns:
        metadata = metadata.rename(columns={"artist": "artist_name"})

    vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.joblib"))

    nn_models = {
        "cosine": joblib.load(os.path.join(DATA_DIR, "nn_cosine.joblib")),
        "euclidean": joblib.load(os.path.join(DATA_DIR, "nn_euclidean.joblib")),
        "manhattan": joblib.load(os.path.join(DATA_DIR, "nn_manhattan.joblib")),
    }

    return metadata, vectors, scaler, nn_models


# ==============================
# RECOMMENDER
# ==============================

def recommend(song_name, metadata, vectors, scaler, nn_model, top_k=5):
    """
    Finds similar songs based on a song name.
    """

    # Find song index
    matches = metadata[metadata["title"].str.lower() == song_name.lower()]

    if len(matches) == 0:
        return ["Song not found. Try another name."]

    idx = matches.index[0]

    # Get vector
    query_vec = vectors[idx].reshape(1, -1)

    # Scale
    query_vec = scaler.transform(query_vec)

    # Nearest neighbors
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_k + 1)

    results = []

    for i in indices[0]:
        if i != idx:  # skip itself
            row = metadata.iloc[i]
            results.append(f"{row['title']} - {row.get('artist_name', 'Unknown')}")

    return results[:top_k]


# ==============================
# UI
# ==============================

st.set_page_config(page_title="Song Recommender", layout="centered")

st.title("🎵 Song Recommender")
st.write("Type a song name to get similar songs.")

# Load data
metadata, vectors, scaler, nn_models = load_assets()

# Metric selector
metric = st.selectbox("Similarity metric", ["cosine", "euclidean", "manhattan"])

# Input
query = st.text_input("Enter song name")

# Run
if query:
    with st.spinner("Finding recommendations..."):
        results = recommend(
            query,
            metadata,
            vectors,
            scaler,
            nn_models[metric]
        )

    st.subheader("Recommendations")
    for i, r in enumerate(results, 1):
        st.write(f"{i}. {r}")