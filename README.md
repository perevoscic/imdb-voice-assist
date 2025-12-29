# IMDB Voice Assistant

Gen-AI conversational voice agent over the IMDB Top 1000 dataset. Supports text + voice input, semantic plot search, and recommendations.

## Stack

- UI: Streamlit
- Chat model: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`
- Speech-to-text: OpenAI Whisper (`whisper-1`)
- Text-to-speech: OpenAI TTS (`tts-1`)

## Project Structure

- `app.py` - Streamlit app
- `imdb_top_1000.csv` - Dataset
- `build_index.py` - Builds local embeddings file
- `data/imdb_embeddings.npz` - Local vector store (generated)
- `data/imdb_embeddings_meta.json` - Vector store metadata (generated)

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your OpenAI key:
   ```bash
   cp .env.example .env
   ```

## Build the Vector Store

Run one of the following:

- From CLI:
  ```bash
  python3 build_index.py
  ```
- Or click **Build embeddings** inside the Streamlit app.

This generates:

- `data/imdb_embeddings.npz`
- `data/imdb_embeddings_meta.json`

If you are sharing via GitHub, commit these generated files so reviewers can test without rebuilding.

## Run the App

```bash
streamlit run app.py
```

## Example Questions

- When did The Matrix release?
- What are the top 5 movies of 2019 by meta score?
- Top 7 comedy movies between 2010-2020 by IMDB rating?
- Top horror movies with a meta score above 85 and IMDB rating above 8
- Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice
- Top 10 movies with over 1M votes but lower gross earnings
- List of movies from the comedy genre where there is death or dead people involved
- Summarize the movie plots of Steven Spielberg’s top-rated sci-fi movies
- List of movies before 1990 that have involvement of police in the plot

## Notes

- Voice input uses `st.audio_input`; allow microphone access in your browser.
- Recommendations are derived from embedding similarity to top results.
- If embeddings are missing, the app will prompt to build them.
