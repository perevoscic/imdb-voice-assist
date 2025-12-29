import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.config import DATA_PATH, FAISS_INDEX_PATH, METADATA_PATH
from src.hybrid_engine import hybrid_search
from src.ingest import load_imdb
from src.query_router import QueryRouter
from src.recommender import recommend
from src.response_generator import ResponseGenerator
from src.semantic_engine import SemanticEngine
from src.speech import SpeechClient
from src.structured_engine import StructuredQuery, run_structured_query

DISPLAY_FIELDS = [
    "Poster_Link",
    "Series_Title",
    "Released_Year",
    "Certificate",
    "Runtime",
    "Genre",
    "IMDB_Rating",
    "Meta_score",
    "Director",
    "No_of_votes",
    "Gross",
    "Overview",
]


def render_movie(movie: Dict[str, Any]) -> None:
    cols = st.columns([1, 3])
    with cols[0]:
        poster = movie.get("Poster_Link")
        if isinstance(poster, str) and poster.startswith("http"):
            st.image(poster, use_container_width=True)
    with cols[1]:
        title = movie.get("Series_Title", "")
        year = movie.get("Released_Year")
        try:
            year_display = str(int(year)) if year is not None else "N/A"
        except (TypeError, ValueError):
            year_display = "N/A"
        st.markdown(f"### {title} ({year_display})")
        st.write(
            f"Certificate: {movie.get('Certificate','N/A')}  |  "
            f"Runtime: {movie.get('Runtime','N/A')}"
        )
        st.write(f"Genre: {movie.get('Genre','N/A')}")
        st.write(f"Director: {movie.get('Director','N/A')}")
        st.write(
            f"IMDb: {movie.get('IMDB_Rating','N/A')}  |  "
            f"Meta: {movie.get('Meta_score','N/A')}"
        )
        st.write(
            f"Votes: {movie.get('No_of_votes','N/A')}  |  "
            f"Gross: {movie.get('Gross','N/A')}"
        )
        with st.expander("Overview"):
            st.write(movie.get("Overview", ""))


st.set_page_config(page_title="IMDB Voice Assistant", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Voice Assistant")

st.markdown(
    """
<style>
div.block-container { padding-bottom: 7rem; }
div[data-testid="stForm"] {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 999;
  background: #0e1117;
  border-top: 1px solid #2a2a2a;
  padding: 0.75rem 1rem 1rem;
}
div[data-testid="stForm"] > div {
  max-width: 900px;
  margin: 0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Options")
    show_reasoning = st.toggle("Show reasoning", value=False)
    show_table = st.toggle("Show matching rows", value=False)
    enable_tts = st.toggle("Enable TTS", value=False)
    st.caption("Data path: " + str(DATA_PATH))

if not DATA_PATH.exists():
    st.warning("Dataset not found. Add data/imdb.csv before running.")

@st.cache_resource
def load_services():
    df = load_imdb()
    router = QueryRouter()
    response_generator = ResponseGenerator()
    speech = SpeechClient()
    semantic_engine = None
    if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
        semantic_engine = SemanticEngine()
    return df, router, response_generator, speech, semantic_engine


df, router, response_generator, speech, semantic_engine = load_services()

def _append_message(
    role: str, content: str, payload: Optional[Dict[str, Any]] = None
) -> None:
    message = {"role": role, "content": content}
    if payload:
        message.update(payload)
    st.session_state.messages.append(message)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if show_reasoning and msg.get("reasoning"):
                st.caption("Reasoning: " + "; ".join(msg["reasoning"]))
            if show_table and msg.get("results"):
                st.dataframe(pd.DataFrame(msg["results"]))
            if msg.get("results"):
                for movie in msg["results"][:5]:
                    render_movie(movie)
            if msg.get("recommendations"):
                st.subheader("Similar movies")
                for movie in msg["recommendations"][:5]:
                    render_movie(movie)

if "last_audio_sig" not in st.session_state:
    st.session_state.last_audio_sig = None

def _run_query(user_text: str) -> None:
    _append_message("user", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)

    router_out = router.route(user_text)

    reasoning: List[str] = []
    results: List[Dict[str, Any]] = []

    if router_out.intent == "semantic":
        if not semantic_engine:
            st.error("Semantic index missing. Run `python -m src.build_index`.")
            st.stop()
        semantic_query = router_out.semantic_query or user_text
        semantic_results = semantic_engine.search(semantic_query, k=5)
        reasoning.append("semantic similarity over Overview")
        for record, score, idx in semantic_results:
            record = dict(record)
            record["similarity"] = score
            record["_row_index"] = idx
            results.append(record)

    elif router_out.intent == "hybrid":
        if not semantic_engine:
            st.error("Semantic index missing. Run `python -m src.build_index`.")
            st.stop()
        structured_query = StructuredQuery(
            filters=router_out.filters,
            sort=router_out.sort,
            limit=router_out.limit,
            groupby=router_out.groupby,
        )
        semantic_query = router_out.semantic_query or user_text
        results, reasoning = hybrid_search(
            df, semantic_engine, structured_query, semantic_query, k=5
        )

    else:
        structured_query = StructuredQuery(
            filters=router_out.filters,
            sort=router_out.sort,
            limit=router_out.limit,
            groupby=router_out.groupby,
        )
        structured_df, reasoning = run_structured_query(df, structured_query)
        for idx, row in structured_df.iterrows():
            record = row.to_dict()
            record["_row_index"] = int(idx)
            results.append(record)

    recommendations = []
    if semantic_engine and results:
        recommendations = recommend(semantic_engine, results, k=5)

    assistant_text = response_generator.generate(
        user_text, results, reasoning, recommendations
    )

    payload = {
        "results": results,
        "recommendations": recommendations,
        "reasoning": reasoning,
    }
    _append_message("assistant", assistant_text, payload=payload)

    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        if show_reasoning and reasoning:
            st.caption("Reasoning: " + "; ".join(reasoning))
        if show_table and results:
            st.dataframe(pd.DataFrame(results))
        if results:
            for movie in results[:5]:
                render_movie(movie)
        if recommendations:
            st.subheader("Similar movies")
            for movie in recommendations[:5]:
                render_movie(movie)

    if enable_tts:
        audio_bytes = speech.synthesize(assistant_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

with st.form("chat_form", clear_on_submit=False):
    cols = st.columns([8, 1, 1])
    user_text = cols[0].text_input(
        "Message",
        key="input_text",
        placeholder="Ask about movies, directors, ratings, plots...",
        label_visibility="collapsed",
    )
    audio_input = None
    if hasattr(st, "audio_input"):
        audio_input = cols[1].audio_input(
            "Mic", key="audio_input", label_visibility="collapsed"
        )
    send_clicked = cols[2].form_submit_button("➤", use_container_width=True)

if send_clicked:
    text = st.session_state.get("input_text", "").strip()
    if text:
        _run_query(text)
        st.session_state.input_text = ""
    else:
        st.warning("Please enter a question or use the mic.")

if audio_input is not None and hasattr(audio_input, "read"):
    audio_bytes = audio_input.read()
    if audio_bytes:
        sig = hashlib.md5(audio_bytes).hexdigest()
        if sig != st.session_state.last_audio_sig:
            st.session_state.last_audio_sig = sig
            transcript = speech.transcribe(audio_bytes, filename="voice.wav")
            if transcript:
                _run_query(transcript)
