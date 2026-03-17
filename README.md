# Song recommender (Million Song Dataset)

Content-based music recommender using the **Million Song Dataset** (MSD). Given a song from the catalogue, it returns 5 similar songs based on audio features (tempo, key, mode, loudness, duration, energy, danceability). No vector DB required: index is stored on disk and queried in memory.

## Project structure

```
song-recommender/
├── config/           # Settings (paths, feature list)
│   └── settings.py
├── src/
│   ├── data/         # Load MSD from HDF5
│   ├── index/        # Build and persist index (scaler + nearest neighbours)
│   └── recommend/    # Query by song name, format output
├── scripts/
│   ├── build_index.py   # Build index from MSD data
│   └── recommend.py     # Get 5 recommendations by song name
├── data/             # Put MSD .h5 files here (not in git)
├── index/            # Built index (not in git)
├── requirements.txt
└── README.md
```

## Setup

1. **Clone and install**

   ```bash
   cd song-recommender
   pip install -r requirements.txt
   ```

2. **Download MSD 1% subset**

   - Get the [Million Song Subset](http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset#subset) (~1.8 GB).
   - Extract it so that `.h5` files live under a single data directory (e.g. `data/` or `MillionSongSubset/data/`).

3. **Build the index**

   ```bash
   python scripts/build_index.py --data-dir path/to/your/msd/data --index-dir index
   ```

   If you put the MSD files in `data/`, you can omit the paths:

   ```bash
   python scripts/build_index.py
   ```

4. **Get recommendations**

   ```bash
   python scripts/recommend.py "Never Gonna Give You Up"
   python scripts/recommend.py "Shape of You" --artist "Ed Sheeran"
   ```

   Or run without arguments for interactive mode:

   ```bash
   python scripts/recommend.py
   ```

   The query song **must** be in the index (i.e. one of the loaded MSD tracks). If it is not found, the script asks you to pick another song.

## Output

Example:

```
Here are your 5 recommendations:

  1. "Song A" — Artist A (genre)
  2. "Song B" — Artist B
  3. "Song C" — Artist C (genre)
  ...
```

## Data source

- [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) (LabROSA / Echo Nest).
- Code for reading MSD HDF5: [MSongsDB](https://github.com/tbertinmahieux/MSongsDB) (PyTables-based schema).

## Licence

Project code: your choice. MSD data and MSongsDB code have their own licences (see MSD website and MSongsDB repo).
