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

Configure environment — create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Required variable:
- `OPENAI_API_KEY` – your OpenAI API key (never commit this)

Optional overrides (sensible defaults are used):
- `OPENAI_MODEL` – chat model (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` – embedding model (default: `text-embedding-3-small`)
- `OPENAI_STT_MODEL` – speech-to-text (default: `whisper-1`)
- `OPENAI_TTS_MODEL` – text-to-speech (default: `tts-1`)

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
- If those files are missing, semantic queries will be unavailable. For a GitHub link, either commit `vectorstore/` or run `python -m src.build_index`.
- For a zip delivery, include `vectorstore/` alongside the codebase.
- Voice features require OpenAI audio endpoints.
