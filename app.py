import os
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Import your existing logic
from src.index import load_index
from src.recommend import recommend, format_recommendations

# --- 1. SETUP PATHS & DATA ---
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT  # Files are in root
SUMMARY_PATH = PROJECT_ROOT / "data" / "msd_summary_file.h5"

print("🚀 Loading Index into memory...")
# In Dash, we load data globally ONCE at startup. 
# It stays in RAM and is shared by all users.
scaler, nn_by_metric, vectors, metadata = load_index(INDEX_PATH)

# --- 2. INITIALIZE APP ---
# We use Bootstrap for a clean layout similar to Streamlit
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Needed for deployment

# --- 3. LAYOUT ---
app.layout = dbc.Container([
    html.H1("🎶 Instant Music Recommender", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("🎵 Enter a song title:"),
            dbc.Input(id="song-input", type="text", placeholder="e.g. Bohemian Rhapsody"),
        ], width=6),
        dbc.Col([
            dbc.Label("🎤 Artist name (optional):"),
            dbc.Input(id="artist-input", type="text", placeholder="e.g. Queen"),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Button("🚀 Recommend Similar Songs", id="recommend-btn", color="primary", className="w-100"),
        ])
    ]),

    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading( # Shows a spinner while the callback runs
                id="loading-output",
                type="default",
                children=html.Pre(id="recommendation-output", style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "padding": "15px"})
            )
        ])
    ])
], str_uid="main-container", style={"maxWidth": "800px"})

# --- 4. CALLBACK (The Logic) ---
@callback(
    Output("recommendation-output", "children"),
    Input("recommend-btn", "n_clicks"),
    State("song-input", "value"),
    State("artist-input", "value"),
    prevent_initial_call=True
)
def update_recommendations(n_clicks, song_title, artist_name):
    if not song_title:
        return "Please enter a song title to begin."

    # Run your original recommendation logic
    recs_by_metric, err = recommend(
        scaler, nn_by_metric, vectors, metadata,
        song_name=song_title,
        artist_name=artist_name if artist_name and artist_name.strip() else None,
        top_k=5
    )

    if err:
        return f"⚠️ {err}"
    
    return format_recommendations(recs_by_metric)

if __name__ == "__main__":
    # Use environment port for cloud deployment
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)