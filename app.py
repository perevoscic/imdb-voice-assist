import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from imdb_data import ImdbConfig, load_imdb_data
from indexing import build_embeddings
from query_engine import needs_al_pacino_clarification, run_query
from response_builder import build_response
from vector_store import VectorStorePaths, load_meta, load_vector_store, save_meta, save_vector_store
from voice import synthesize_speech, transcribe_audio


load_dotenv()


@st.cache_data(show_spinner=False)
def get_data():
    config = ImdbConfig()
    df = load_imdb_data(config)
    return df


@st.cache_resource(show_spinner=False)
def get_vector_store() -> tuple:
    paths = VectorStorePaths()
    embeddings, row_ids = load_vector_store(paths)
    meta = load_meta(paths)
    return embeddings, row_ids, meta


def build_vector_store(client: OpenAI):
    df = get_data()
    texts = df["Overview"].fillna("").astype(str).tolist()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = build_embeddings(client, texts, model=model)
    row_ids = df.index.tolist()
    paths = VectorStorePaths()
    save_vector_store(embeddings, row_ids, paths)
    save_meta(
        {
            "rows": len(row_ids),
            "model": model,
            "source": ImdbConfig().csv_path,
        },
        paths,
    )
    st.cache_resource.clear()


def resolve_input(audio_data, text_input: Optional[str], client: OpenAI) -> Optional[str]:
    if audio_data is not None:
        transcript = transcribe_audio(client, audio_data.read())
        return transcript
    return text_input


def main():
    st.set_page_config(page_title="IMDB Voice Assistant", page_icon="🎬", layout="wide")
    st.title("IMDB Voice Assistant")
    st.caption("Ask movie questions with voice or text. Powered by GPT-4o-mini + IMDB Top 1000.")

    client = OpenAI()

    df = get_data()
    embeddings, row_ids, meta = get_vector_store()

    if embeddings.size == 0:
        st.warning("Embeddings not found. Build the local vector store to enable semantic search.")
        if st.button("Build embeddings"):
            with st.spinner("Building embeddings..."):
                build_vector_store(client)
            st.success("Embeddings built. Reloaded vector store.")
            embeddings, row_ids, meta = get_vector_store()

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Chat model", value=os.getenv("CHAT_MODEL", "gpt-4o-mini"), disabled=True)
        st.text_input(
            "Embedding model",
            value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            disabled=True,
        )
        st.text_input("STT model", value=os.getenv("STT_MODEL", "whisper-1"), disabled=True)
        st.text_input("TTS model", value=os.getenv("TTS_MODEL", "tts-1"), disabled=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_pacino" not in st.session_state:
        st.session_state.pending_pacino = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

    audio_input = st.audio_input("Speak a question")
    user_input = st.chat_input("Ask about movies, directors, ratings, plot, etc.")

    query = resolve_input(audio_input, user_input, client)

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.pending_pacino:
            clarified = query.lower()
            role_hint = "lead" if "lead" in clarified or "star1" in clarified else "any role"
            query = f"{st.session_state.pending_query} (Al Pacino as {role_hint})"
            st.session_state.pending_pacino = False
            st.session_state.pending_query = ""

        if needs_al_pacino_clarification(query):
            follow_up = (
                "Are you looking for movies where Al Pacino is the lead actor (Star1), "
                "or any movie where he appears in any role?"
            )
            st.session_state.pending_pacino = True
            st.session_state.pending_query = query
            st.session_state.messages.append({"role": "assistant", "content": follow_up})
            with st.chat_message("assistant"):
                st.markdown(follow_up)
            return

        with st.spinner("Thinking..."):
            _, results, recommendations = run_query(df, embeddings, row_ids, client, query)
            response_text = build_response(client, query, results, recommendations)

        audio_response = synthesize_speech(client, response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "audio": audio_response}
        )
        with st.chat_message("assistant"):
            st.markdown(response_text)
            if audio_response:
                st.audio(audio_response, format="audio/mp3")

        if results:
            st.subheader("Results")
            for item in results:
                poster = item.get("Poster_Link")
                title = item.get("Series_Title", "Movie")
                if poster:
                    st.image(poster, width=140, caption=title)
            st.dataframe(results, use_container_width=True)
        if recommendations:
            st.subheader("Recommendations")
            for item in recommendations:
                poster = item.get("Poster_Link")
                title = item.get("Series_Title", "Movie")
                if poster:
                    st.image(poster, width=120, caption=title)
            st.dataframe(recommendations, use_container_width=True)


if __name__ == "__main__":
    main()
