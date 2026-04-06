import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from src.recommend.query import recommend, format_recommendations
from src.index import load_index

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
        "cosine": joblib.load(os.path.join(DATA_DIR, "nn_cosine.joblib")),
        "euclidean": joblib.load(os.path.join(DATA_DIR, "nn_euclidean.joblib")),
        "manhattan": joblib.load(os.path.join(DATA_DIR, "nn_manhattan.joblib")),
    }

    return metadata, vectors, scaler, nn_models

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
            recs_by_metric, err = recommend(
            nn_models, vectors, metadata,
            query_song,
            query_artist,
            top_k=1,
        )
        if err:
            st.error(err)
        else:
            cols = st.columns(len(recs_by_metric))
            for col, (metric, recs) in zip(cols, recs_by_metric.items()):
                with col:
                    st.subheader(metric.capitalize())
                    for _, row in recs.iterrows():
                        genre = row.get("genre", "")
                        if pd.notna(genre) and str(genre).strip():
                            st.markdown(f"**{row['title']}** — {row['artist_name']} ({genre})")
                        else:
                            st.markdown(f"**{row['title']}** — {row['artist_name']}")