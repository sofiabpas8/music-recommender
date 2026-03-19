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

    # Rename columns
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
# HELPER
# ==============================

def normalize(text):
    return str(text).lower().replace("the ", "").strip()

# ==============================
# RECOMMENDER
# ==============================

def recommend(song_name, metadata, vectors, scaler, nn_model, artist_name=None, top_k=1):
    matches = metadata[metadata["title"].str.lower().str.contains(song_name.lower())]
    if artist_name:
        artist_matches = matches[matches["artist_name"].str.lower().str.contains(artist_name.lower())]

    if len(artist_matches) > 0:
        matches = artist_matches

    # --- No results ---
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

st.set_page_config(page_title="🎵 Song Recommender", layout="wide")
st.title("🎵 Song Recommender")
st.markdown(
    "Before searching for a song, please read the instructions provided in this "
    "[Instruction Form](https://forms.gle/VjB1wjW2EcP9272KA) "
    "and then fill the questionnaire."
)

# Load data
metadata, vectors, scaler, nn_models = load_assets()

# Inputs in two columns
col1, col2 = st.columns([3, 1])
with col1:
    query_song = st.text_input("Enter song name")
    query_artist = st.text_input("Enter artist name (optional)")
with col2:
    search_clicked = st.button("Search")

st.markdown("---")

# Run recommendations
if search_clicked:
    if not query_song:
        st.warning("Please enter a song name.")
    else:
        with st.spinner("Finding recommendations..."):
            # Use columns to display metrics side by side
            cols = st.columns(len(nn_models))
            for idx, (metric_name, model) in enumerate(nn_models.items()):
                with cols[idx]:
                    st.subheader(f"{metric_name}")
                    results = recommend(query_song, metadata, vectors, scaler, model, artist_name=query_artist)
                    for i, r in enumerate(results, 1):
                        st.markdown(f"**{i}.** {r}")