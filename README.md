# IMDB Voice Assistant

Gen-AI conversational voice agent over the IMDB Top 1000 dataset. Supports text + voice input, semantic plot search, and recommendations.

## Stack

- UI: Streamlit
- Realtime voice: `streamlit-webrtc`
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

## Safety & Scope

Policy (simple):
- Allowed: movie questions, recommendations, plot summaries, dataset stats
- Disallowed: hate/harassment, sexual content involving minors, self-harm, instructions for wrongdoing, doxxing, explicit porn
- Profanity alone is allowed (warn + continue); blocking only when it targets a person or group

Runtime enforcement:
- Moderation gate runs before any retrieval or LLM calls, including voice transcripts (OpenAI moderation endpoint by default)
- Out-of-scope requests are redirected: "I'm an IMDb assistant. Ask about movies, directors, ratings, years, genres, plots."
- Strict JSON validation for LLM-planned queries with Pydantic
- Rate limiting (per-session) with exponential backoff and max input size limits
- No raw audio or secrets in logs; API keys stay in env vars
 - If swapping to Gemini, apply safety settings and block or redirect on threshold

Implementation notes:
- `src/safety.py` handles moderation, scope checks, and safety messages
- `app.py` calls safety checks before `check_needs_clarification` or `run_query`
- Optional env vars: `MODERATION_MODEL`, `SCOPE_MODEL`, `MAX_INPUT_CHARS`, `MAX_AUDIO_BYTES`, `RATE_LIMIT_PER_MINUTE`, `OPENAI_TIMEOUT`

## Notes

- Voice input uses `st.audio_input`; allow microphone access in your browser.
- Realtime voice streaming uses `streamlit-webrtc`; allow microphone access and pick **Realtime (WebRTC)** in the sidebar.
- Recommendations are derived from embedding similarity to top results.
- If embeddings are missing, the app will prompt to build them.
