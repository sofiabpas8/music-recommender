from taipy.gui import Gui, notify
from pathlib import Path
from src.index import load_index
from src.recommend import recommend, format_recommendations

# --- 1. SETUP PATHS & LOAD DATA ---
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "index"
SUMMARY_PATH = PROJECT_ROOT / "data" / "msd_summary_file.h5"

print("Initializing Recommender... This may take a moment.")
scaler, nn_by_metric, vectors, metadata = load_index(INDEX_PATH)

# --- 2. APP STATE ---
selected_song = ""
artist_name = ""
recommendations_output = ""
show_results = False

def handle_recommend(state):
    """Function triggered by the button click."""
    if not state.selected_song.strip():
        notify(state, "warning", "Please enter a song title.")
        return

    # Notify user that processing has started
    notify(state, "info", "Searching for recommendations...")
    state.show_results = False
    
    recs_by_metric, err = recommend(
        scaler,
        nn_by_metric,
        vectors,
        metadata,
        song_name=state.selected_song,
        artist_name=state.artist_name if state.artist_name.strip() else None,
        top_k=5
    )

    if err:
        state.recommendations_output = f"⚠️ {err}"
        notify(state, "error", err)
    else:
        # Format the results for display
        state.recommendations_output = format_recommendations(recs_by_metric)
        state.show_results = True
        notify(state, "success", "Recommendations ready!")

# --- 3. USER INTERFACE (Markdown) ---
# <|{variable}|input|> creates a reactive binding
page = """
# 🎶 Instant Music Recommender

<|layout|columns=1 1|gap=20px|
<|{selected_song}|input|label=🎵 Enter a song title:|class_name=fullwidth|>
<|{artist_name}|input|label=🎤 Artist name (optional):|class_name=fullwidth|>
|>

<br/>
<center>
<|Recommend Similar Songs|button|on_action=handle_recommend|class_name=active|>
</center>

---

### 🎶 Recommendations:
<|{recommendations_output}|text|raw=True|>
"""

if __name__ == "__main__":
    # run() starts the web server
    Gui(page=page).run(
        title="Music Recommender 🎧",
        dark_mode=True,
        port=8080
    )