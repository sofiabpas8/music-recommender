import os
import sys
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# --- 1. SETUP PATHS & DATA ---
# BASE_DIR is the root folder (where scaler.joblib and metadata.pkl sit)
BASE_DIR = Path(__file__).resolve().parent

# We add 'src' to the system path so Python can find your custom modules
sys.path.append(str(BASE_DIR))

# Import your existing logic from the src folder
try:
    from src.index import load_index
    from src.recommend import recommend, format_recommendations
    print("✅ Successfully imported recommendation modules.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current Path: {sys.path}")
    sys.exit(1)

# --- 2. LOAD DATA (Global Scope) ---
# This runs once when the server starts and is shared by all users
print("🚀 Loading Index into memory (this may take a moment)...")
try:
    # We pass BASE_DIR because your .joblib/.pkl files are in the project root
    scaler, nn_by_metric, vectors, metadata = load_index(BASE_DIR)
    print("✅ Index loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {e}")
    print(f"Files detected in root: {os.listdir(BASE_DIR)}")
    # Exit so Render doesn't hang on a broken process
    sys.exit(1)

# --- 3. INITIALIZE DASH APP ---
# We use the Cyborg theme for a dark, "music-app" feel
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # Gunicorn looks for this 'server' variable

# --- 4. APP LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🎧 Music Recommender", className="text-center mt-5 mb-4"),
            html.P("Find your next favorite song based on the Million Song Dataset.", 
                   className="text-center text-muted mb-5"),
            
            dbc.Card([
                dbc.CardBody([
                    dbc.Label("🎵 Song Title"),
                    dbc.Input(id="song-input", type="text", placeholder="Enter a song name...", className="mb-3"),
                    
                    dbc.Label("🎤 Artist Name (Optional)"),
                    dbc.Input(id="artist-input", type="text", placeholder="Enter artist...", className="mb-4"),
                    
                    dbc.Button("🚀 Get Recommendations", id="recommend-btn", color="info", className="w-100"),
                ])
            ], className="shadow-sm"),
            
            html.Hr(className="my-5"),
            
            # Loading spinner wraps the output area
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div(id="recommendation-output")
            )
        ], width={"size": 8, "offset": 2})
    ])
], fluid=True)

# --- 5. CALLBACK LOGIC ---
@callback(
    Output("recommendation-output", "children"),
    Input("recommend-btn", "n_clicks"),
    State("song-input", "value"),
    State("artist-input", "value"),
    prevent_initial_call=True
)
def update_output(n_clicks, song_title, artist_name):
    if not song_title or not song_title.strip():
        return dbc.Alert("Please enter a song title to search.", color="warning")

    # Call your src/recommend.py logic
    recs_by_metric, err = recommend(
        scaler, 
        nn_by_metric, 
        vectors, 
        metadata, 
        song_name=song_title.strip(),
        artist_name=artist_name.strip() if artist_name and artist_name.strip() else None,
        top_k=5
    )

    if err:
        return dbc.Alert(f"Error: {err}", color="danger")
    
    # Format the string results
    formatted_results = format_recommendations(recs_by_metric)
    
    return html.Div([
        html.H4("🎶 We recommend:", className="mb-3"),
        html.Pre(formatted_results, style={
            "backgroundColor": "#1a1a1a", 
            "padding": "20px", 
            "borderRadius": "5px",
            "color": "#00d1b2",
            "fontSize": "14px"
        })
    ])

# --- 6. RUN SERVER ---
if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8050))
    # host 0.0.0.0 is required for the cloud to access the container
    app.run_server(debug=False, host="0.0.0.0", port=port)