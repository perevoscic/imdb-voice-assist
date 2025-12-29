# IMDB Voice Assistant

A local-first GenAI voice agent that answers questions about the IMDB dataset using structured retrieval plus semantic similarity search over plot overviews.

## Features
- Structured filters and ranking (year ranges, ratings, votes, gross, director aggregates)
- Semantic similarity search on `Overview` for plot-based queries
- Hybrid mode (filters + similarity)
- Streamlit chat UI with audio input, transcription, and optional TTS
- Optional result table and reasoning display

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add your key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Data
Place the IMDB dataset CSV at `data/imdb.csv`.

## Build the vector index (one-time)

```bash
python -m src.build_index
```

## Run the app

```bash
streamlit run app.py
```

## Notes
- The local vector store files are saved to `vectorstore/imdb_faiss.index` and `vectorstore/imdb_metadata.pkl`.
- If those files are missing, semantic queries will be unavailable.
- Voice features require OpenAI audio endpoints.
