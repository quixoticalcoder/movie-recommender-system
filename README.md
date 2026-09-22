# movie-recommender-system

A content-based movie recommendation prototype with an offline metadata pipeline and a Streamlit interface. Select a movie to retrieve up to five titles ranked by cosine similarity over bag-of-words features. Recommendations use movie metadata, rather than user ratings or viewing history.

## Architecture

```text
movie.csv + credits.csv
          │ merge on title; extract metadata
          ▼
Overview + genres + keywords + first three cast members + director
          │ lowercase; CountVectorizer
          ▼
Up to 5,000 count features → dense cosine similarity matrix
          │ pickle serialization
          ▼
Streamlit title selector → ranked movie titles
```

`CountVectorizer` removes English stop words. For feature vectors x and y, similarity is `x · y / (‖x‖ ‖y‖)`. The interface displays titles only; it does not fetch posters or call a recommendation API. See the implementation limitations below for preprocessing caveats.

## Repository layout

| File | Responsibility |
| --- | --- |
| `Movierecommender.py` | Prepare metadata, compute similarities, and serialize artifacts |
| `app.py` | Load artifacts and serve the recommendation interface |
| `requirements.txt` | Direct Python dependencies imported by the scripts |

## Local setup

Use Python 3.11 as a starting environment. Dependencies are unpinned; this repository does not provide a reproducible version lock.

```bash
git clone https://github.com/quixoticalcoder/movie-recommender-system.git
cd movie-recommender-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

### Prepare the input data

Supply your own compatible, appropriately licensed metadata CSVs at the paths currently hardcoded in the training script:

| Path | Required columns |
| --- | --- |
| `.venv/movie.csv` | `id`, `title`, `overview`, `genres`, `keywords` |
| `.venv/credits.csv` | `title`, `cast`, `crew` |

The metadata fields `genres`, `keywords`, `cast`, and `crew` must contain string representations of lists of dictionaries accepted by `ast.literal_eval`. Entries use a `name` key; crew entries also require `job`, with `Director` selecting the director. Preserve cast ordering because only the first three entries are retained. Avoid overlapping columns beyond `title`, which would receive merge suffixes.

The CSVs and generated artifacts are not tracked. Keep the CSVs outside version control; recreating `.venv` will also remove data stored there.

### Generate artifacts and start the app

Run both commands from the repository root:

```bash
python Movierecommender.py
python -m streamlit run app.py
```

The first command writes `movie.dict.pkl`, `similarity.pkl`, and `movies.pkl` to the current directory. The app consumes the first two; regenerate them together so movie ordering matches matrix rows. `movies.pkl` is an additional DataFrame export. Only load pickle files you generated or trust: deserialization can execute code.

Select a title in the browser and click **Recommend**. No API key is needed.

## Implementation boundaries

- **Keyword preprocessing:** parsed keywords are printed but not assigned back. The subsequent transformation iterates over the original string, so keyword names do not enter the vocabulary as intended.
- **Stemming order:** Porter stemming runs after vectorization. Serialized tags are stemmed, but the similarity matrix uses the earlier, unstemmed features.
- **Row alignment:** dropping incomplete rows preserves DataFrame index labels. Recommendation lookup treats those labels as matrix positions, which can produce incorrect results or an index error when rows were dropped.
- **Demo prerequisite:** training calls `recommend('Batman Begins')` before writing artifacts. The merged data must contain that exact title for the script to finish.
- **Ranking assumptions:** lookup selects the first matching title; sorting discards the first ranked item as the presumed query. Duplicate titles or tied feature vectors can violate those assumptions.
- **Scale and validation:** feature vectors and the pairwise matrix are dense; matrix storage grows quadratically with movie count. No held-out quality evaluation, personalization, or automated test suite is included.

These constraints describe the current implementation and should be addressed before treating it as a production recommendation service.
