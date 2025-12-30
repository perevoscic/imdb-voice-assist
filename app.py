import hashlib
import io
import logging
import os
import queue
import time
import wave
from typing import Optional

import tornado.ioloop
import tornado.iostream
import tornado.websocket

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from imdb_data import ImdbConfig, load_imdb_data
from indexing import build_embeddings
from query_engine import check_needs_clarification, run_query
from response_builder import build_response
from src.safety import (
    contains_profanity,
    is_in_scope,
    moderate_text,
    out_of_scope_message,
    profanity_nudge_message,
    safety_refusal_message,
)
from vector_store import VectorStorePaths, load_meta, load_vector_store, save_meta, save_vector_store
from voice import clean_text_for_speech, synthesize_speech, transcribe_audio


load_dotenv()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(5 * 1024 * 1024)))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
logger = logging.getLogger(__name__)


def configure_asyncio_exception_handler():
    """Suppress expected WebSocket close errors from Tornado/Streamlit tasks."""
    if os.getenv("SUPPRESS_WS_CLOSE_ERRORS", "1") != "1":
        return
    try:
        loop = tornado.ioloop.IOLoop.current().asyncio_loop
    except Exception:
        return

    logged = False
    log_suppressed = os.getenv("LOG_SUPPRESSED_WS_ERRORS", "0") == "1"

    def _handler(loop, context):
        nonlocal logged
        exc = context.get("exception")
        if isinstance(
            exc,
            (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError),
        ):
            if log_suppressed and not logged:
                logger.info("Suppressed WebSocket close error from Tornado.")
                logged = True
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


def load_css():
    """Load CSS from external file."""
    css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_hero():
    """Render the hero header section with new chat button."""
    st.markdown("""
    <div class="hero-section">
        <span class="hero-emoji">🎬</span>
        <p class="hero-title-text">Movie Voice Assistant</p>
        <p class="hero-subtitle-text">Ask me anything about movies using your voice or text. I know the top 1000 IMDB films inside out.</p>
    </div>
    """, unsafe_allow_html=True)


def clear_conversation():
    """Clear the conversation state without page reload."""
    st.session_state.messages = []
    st.session_state.pending_clarification = False
    st.session_state.pending_query = ""
    st.session_state.clarification_question = ""
    st.session_state.pending_voice_text = ""
    st.session_state.last_audio_hash = ""
    st.session_state.live_transcript = ""
    st.session_state.audio_frame_buffer = []
    st.session_state.audio_buffer_samples = 0


def render_empty_state():
    """Render the empty state when there are no messages."""
    pass


def render_safety_section():
    st.markdown("### 🛡️ Safety")
    st.caption("Allowed: movie questions, recommendations, plot summaries, dataset stats.")
    st.caption("Not allowed: harassment, self-harm, sexual content involving minors, doxxing, explicit porn.")
    st.caption("Profanity is OK if not targeted at a person or group.")


def check_rate_limit() -> tuple[bool, int]:
    now = time.time()
    window_seconds = 60
    timestamps = st.session_state.get("request_timestamps", [])
    timestamps = [ts for ts in timestamps if now - ts < window_seconds]
    throttle_until = st.session_state.get("throttle_until", 0.0)
    if now < throttle_until:
        retry_after = int(throttle_until - now)
        st.session_state["request_timestamps"] = timestamps
        return False, max(retry_after, 1)
    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        level = min(st.session_state.get("throttle_level", 0) + 1, 6)
        wait_seconds = min(60, 2**level)
        st.session_state["throttle_level"] = level
        st.session_state["throttle_until"] = now + wait_seconds
        st.session_state["request_timestamps"] = timestamps
        return False, wait_seconds
    timestamps.append(now)
    st.session_state["request_timestamps"] = timestamps
    st.session_state["throttle_level"] = max(st.session_state.get("throttle_level", 0) - 1, 0)
    st.session_state["throttle_until"] = 0.0
    return True, 0


def respond_with_text(
    client: OpenAI,
    text: str,
    voice_replies: bool,
    auto_play_audio: bool,
):
    audio_response = None
    if voice_replies:
        speakable_text = clean_text_for_speech(text)
        audio_response = synthesize_speech(client, speakable_text)
    st.session_state.messages.append({"role": "assistant", "content": text, "audio": audio_response})
    with st.chat_message("assistant", avatar="🎬"):
        st.markdown(text)
        if audio_response:
            st.audio(audio_response, format="audio/mp3", autoplay=auto_play_audio)


def show_processing_indicator(status_placeholder, status: str, substatus: str = "", progress: int = 0):
    """Display an animated processing indicator with status updates."""
    icon_map = {
        "Listening": "🎙️",
        "Analyzing": "🔍",
        "Searching": "🎞️",
        "Crafting": "✨",
        "Preparing": "🔊",
        "Done": "✅",
    }
    
    icon = "🎬"
    for key, val in icon_map.items():
        if key in status:
            icon = val
            break
    
    status_placeholder.markdown(f"""
    <div class="processing-container">
        <div class="processing-icon">{icon}</div>
        <div class="wave-container">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        <div class="processing-text">{status}</div>
        <div class="processing-subtext">{substatus}</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {progress}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _data_cache_key(config: ImdbConfig) -> str:
    """Cache-buster tied to the CSV file; updates invalidate cached data."""
    path = config.csv_path
    try:
        stat = os.stat(path)
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    except FileNotFoundError:
        return f"{path}:missing"


@st.cache_data(show_spinner=False)
def get_data(cache_key: str):
    # cache_key ensures Streamlit invalidates the cache when the CSV changes
    _ = cache_key  # silence unused variable lint while keeping cache binding
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
    df = get_data(_data_cache_key(ImdbConfig()))
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


def _suffix_from_mime(mime_type: Optional[str]) -> Optional[str]:
    if not mime_type:
        return None
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
    }
    return mapping.get(mime_type)


def resolve_input(
    audio_bytes: Optional[bytes],
    text_input: Optional[str],
    client: OpenAI,
    status_placeholder=None,
    audio_suffix: Optional[str] = None,
) -> Optional[str]:
    if audio_bytes:
        if status_placeholder:
            show_processing_indicator(
                status_placeholder,
                "Listening to your voice...",
                "Transcribing audio with Whisper AI",
                50
            )
        try:
            transcript = transcribe_audio(client, audio_bytes, suffix=audio_suffix)
        except Exception:
            logger.exception("Voice transcription failed")
            if status_placeholder:
                status_placeholder.empty()
            st.error("Voice transcription failed. Please try again.")
            return None
        if status_placeholder:
            status_placeholder.empty()
        return transcript
    return text_input


def _frames_to_wav_bytes(frames, sample_rate: int) -> Optional[bytes]:
    if not frames or not sample_rate:
        return None
    chunks = []
    for frame in frames:
        data = frame.to_ndarray()
        if data.ndim > 1:
            data = data.mean(axis=0)
        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)
        chunks.append(data)
    if not chunks:
        return None
    samples = np.concatenate(chunks)
    if samples.size == 0:
        return None
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def _format_display(items):
    formatted = []
    for item in items:
        entry = {}
        for key, value in item.items():
            label = key.replace("_", " ")
            entry[label] = value
        gross = entry.get("Gross")
        if isinstance(gross, (int, float)) and not isinstance(gross, bool):
            entry["Gross"] = f"${gross:,.0f}"
        formatted.append(entry)
    return formatted


def render_results_section(results, recommendations):
    """Render the results and recommendations as tables (posters are in AI response)."""
    if results:
        st.markdown("""
        <div class="results-header">
            <span class="icon">📊</span>
            <h3>Full Details</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(_format_display(results), width="stretch")
    
    if recommendations:
        st.markdown("""
        <div class="results-header">
            <span class="icon">💡</span>
            <h3>You Might Also Like</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(_format_display(recommendations), width="stretch")


def main():
    configure_asyncio_exception_handler()
    st.set_page_config(
        page_title="Movie Voice Assistant",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_css()
    
    # Render new chat button (fixed position via CSS using .st-key-new_chat_btn)
    if st.button("✨", key="new_chat_btn", help="Start a new conversation"):
        clear_conversation()
        st.rerun()
    
    # Render hero header
    render_hero()

    client = OpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT", "30")))

    df = get_data(_data_cache_key(ImdbConfig()))
    embeddings, row_ids, meta = get_vector_store()

    if embeddings.size == 0:
        st.warning("⚠️ Embeddings not found. Build the local vector store to enable semantic search.")
        if st.button("🔧 Build Embeddings"):
            with st.spinner("Building embeddings..."):
                build_vector_store(client)
            st.success("✅ Embeddings built successfully!")
            embeddings, row_ids, meta = get_vector_store()

    # Sidebar with settings
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        st.markdown("**Chat Model**")
        st.text_input("chat_model", value=os.getenv("CHAT_MODEL", "gpt-4o-mini"), disabled=True, label_visibility="collapsed")
        
        st.markdown("**Embedding Model**")
        st.text_input("embed_model", value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), disabled=True, label_visibility="collapsed")
        
        st.markdown("**Speech-to-Text**")
        st.text_input("stt_model", value=os.getenv("STT_MODEL", "whisper-1"), disabled=True, label_visibility="collapsed")
        
        st.markdown("**Text-to-Speech**")
        st.text_input("tts_model", value=os.getenv("TTS_MODEL", "tts-1"), disabled=True, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 🎙️ Voice Mode")
        voice_only_mode = st.toggle("Voice-only mode", value=False)
        voice_input_mode = st.radio(
            "Voice input mode",
            options=["Push-to-talk", "Realtime (WebRTC)"],
            index=1 if voice_only_mode else 0
        )
        voice_replies = st.toggle("Voice replies", value=True)
        auto_play_audio = st.toggle("Autoplay voice replies", value=True)
        auto_send_voice = st.toggle("Auto-send voice input", value=True)
        realtime_chunk_seconds = st.slider(
            "Realtime chunk seconds",
            min_value=1.5,
            max_value=5.0,
            value=2.5,
            step=0.5
        )

        if voice_only_mode and voice_input_mode != "Realtime (WebRTC)":
            st.info("Voice-only mode uses realtime streaming.")
            voice_input_mode = "Realtime (WebRTC)"

        st.markdown("---")
        st.markdown(f"📊 **Database:** {meta.get('rows', 0):,} movies indexed")
        st.markdown("---")
        render_safety_section()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_clarification" not in st.session_state:
        st.session_state.pending_clarification = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "clarification_question" not in st.session_state:
        st.session_state.clarification_question = ""
    if "pending_voice_text" not in st.session_state:
        st.session_state.pending_voice_text = ""
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = ""
    if "audio_frame_buffer" not in st.session_state:
        st.session_state.audio_frame_buffer = []
    if "audio_buffer_samples" not in st.session_state:
        st.session_state.audio_buffer_samples = 0
    if "live_transcript" not in st.session_state:
        st.session_state.live_transcript = ""
    if "last_transcribe_ts" not in st.session_state:
        st.session_state.last_transcribe_ts = 0.0
    if "streaming_prev" not in st.session_state:
        st.session_state.streaming_prev = False
    if "stream_sample_rate" not in st.session_state:
        st.session_state.stream_sample_rate = 0

    # Render existing messages or empty state
    if not st.session_state.messages:
        render_empty_state()
    else:
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "🎬"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                if message.get("audio"):
                    st.audio(message["audio"], format="audio/mp3", autoplay=auto_play_audio)

    query = None

    if voice_input_mode == "Realtime (WebRTC)":
        if voice_only_mode:
            st.markdown('<div class="voice-only-title">Voice Conversation</div>', unsafe_allow_html=True)
            st.markdown('<div class="voice-only-subtitle">Tap to start, speak naturally, then stop to send.</div>', unsafe_allow_html=True)

        webrtc_ctx = webrtc_streamer(
            key="realtime-voice",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=512,
            media_stream_constraints={"audio": True, "video": False},
        )

        if webrtc_ctx.audio_receiver:
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.5)
            except queue.Empty:
                frames = []

            if frames:
                st.session_state.stream_sample_rate = frames[0].sample_rate or st.session_state.stream_sample_rate
                st.session_state.audio_frame_buffer.extend(frames)
                st.session_state.audio_buffer_samples += sum(frame.samples for frame in frames)

        sample_rate = st.session_state.stream_sample_rate
        buffered_seconds = (
            st.session_state.audio_buffer_samples / sample_rate if sample_rate else 0.0
        )
        if buffered_seconds >= realtime_chunk_seconds and (time.time() - st.session_state.last_transcribe_ts) > 0.5:
            wav_bytes = _frames_to_wav_bytes(st.session_state.audio_frame_buffer, sample_rate)
            st.session_state.audio_frame_buffer = []
            st.session_state.audio_buffer_samples = 0
            st.session_state.last_transcribe_ts = time.time()
            if wav_bytes:
                try:
                    chunk_text = transcribe_audio(client, wav_bytes, suffix=".wav")
                except Exception:
                    logger.exception("Realtime voice transcription failed")
                    chunk_text = None
                if chunk_text:
                    if st.session_state.live_transcript:
                        st.session_state.live_transcript = f"{st.session_state.live_transcript.strip()} {chunk_text.strip()}"
                    else:
                        st.session_state.live_transcript = chunk_text.strip()

        is_streaming = bool(getattr(webrtc_ctx.state, "playing", False)) if webrtc_ctx else False
        if st.session_state.streaming_prev and not is_streaming:
            wav_bytes = _frames_to_wav_bytes(st.session_state.audio_frame_buffer, sample_rate)
            st.session_state.audio_frame_buffer = []
            st.session_state.audio_buffer_samples = 0
            if wav_bytes:
                try:
                    tail_text = transcribe_audio(client, wav_bytes, suffix=".wav")
                except Exception:
                    logger.exception("Realtime voice transcription failed")
                    tail_text = None
                if tail_text:
                    if st.session_state.live_transcript:
                        st.session_state.live_transcript = f"{st.session_state.live_transcript.strip()} {tail_text.strip()}"
                    else:
                        st.session_state.live_transcript = tail_text.strip()
            if auto_send_voice and st.session_state.live_transcript.strip():
                query = st.session_state.live_transcript.strip()
                st.session_state.live_transcript = ""

        st.session_state.streaming_prev = is_streaming

        transcript_value = st.text_area(
            "Live transcript",
            value=st.session_state.live_transcript,
            height=120,
            placeholder="Your live transcript will appear here..."
        )
        if transcript_value != st.session_state.live_transcript:
            st.session_state.live_transcript = transcript_value

        action_col, clear_col = st.columns([1, 1])
        with action_col:
            if st.button("Send transcript", disabled=not st.session_state.live_transcript.strip()):
                query = st.session_state.live_transcript.strip()
                st.session_state.live_transcript = ""
        with clear_col:
            if st.button("Clear transcript"):
                st.session_state.live_transcript = ""

        if not voice_only_mode:
            user_input = st.chat_input("Ask about movies, directors, ratings, plot...")
            if user_input:
                query = user_input
    else:
        input_col, mic_col = st.columns([7, 1])
        with input_col:
            user_input = st.chat_input("Ask about movies, directors, ratings, plot...")
        with mic_col:
            audio_input = st.audio_input("🎙️", label_visibility="collapsed")

        voice_status = st.empty() if audio_input else None
        audio_bytes = audio_input.read() if audio_input else None
        if audio_bytes and len(audio_bytes) > MAX_AUDIO_BYTES:
            st.warning("Audio clip too large. Please upload a shorter recording.")
            audio_bytes = None
        if audio_bytes:
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                audio_suffix = None
                if getattr(audio_input, "name", None):
                    _, ext = os.path.splitext(audio_input.name)
                    audio_suffix = ext or None
                if not audio_suffix:
                    audio_suffix = _suffix_from_mime(getattr(audio_input, "type", None))
                transcript = resolve_input(
                    audio_bytes,
                    None,
                    client,
                    voice_status,
                    audio_suffix=audio_suffix,
                )
                if transcript:
                    if auto_send_voice:
                        query = transcript
                    else:
                        st.session_state.pending_voice_text = transcript

        if st.session_state.pending_voice_text:
            st.markdown("**Review your voice transcript**")
            edited_transcript = st.text_area(
                "Voice transcript",
                value=st.session_state.pending_voice_text,
                label_visibility="collapsed"
            )
            review_col, discard_col = st.columns([1, 1])
            with review_col:
                if st.button("Send transcript"):
                    query = edited_transcript.strip()
                    st.session_state.pending_voice_text = ""
            with discard_col:
                if st.button("Discard transcript"):
                    st.session_state.pending_voice_text = ""

        if not query:
            query = user_input

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        if len(query) > MAX_INPUT_CHARS:
            respond_with_text(
                client,
                f"That message is too long. Please keep requests under {MAX_INPUT_CHARS} characters.",
                voice_replies,
                auto_play_audio,
            )
            return

        rate_ok, retry_after = check_rate_limit()
        if not rate_ok:
            respond_with_text(
                client,
                f"Too many requests. Please wait {retry_after} seconds and try again.",
                voice_replies,
                auto_play_audio,
            )
            return

        moderation = moderate_text(client, query)
        if moderation.flagged:
            respond_with_text(
                client,
                safety_refusal_message(),
                voice_replies,
                auto_play_audio,
            )
            return

        # Get conversation history for context
        conversation_history = st.session_state.messages[:-1]

        scope_result = is_in_scope(client, query, conversation_history)
        if not scope_result.in_scope:
            respond_with_text(
                client,
                out_of_scope_message(),
                voice_replies,
                auto_play_audio,
            )
            return

        tone_nudge = profanity_nudge_message() if contains_profanity(query) else None

        # Handle pending clarification responses
        if st.session_state.pending_clarification:
            query = f"{st.session_state.pending_query} ({query})"
            st.session_state.pending_clarification = False
            st.session_state.pending_query = ""
            st.session_state.clarification_question = ""

        # Show animated processing indicator
        status_placeholder = st.empty()
        
        # Phase 1: Analyzing
        show_processing_indicator(
            status_placeholder,
            "Analyzing your question...",
            "Understanding your intent",
            15
        )
        
        # Check if clarification is needed
        clarification = check_needs_clarification(client, query, conversation_history, df)
        
        if clarification:
            if tone_nudge:
                clarification = f"{tone_nudge}\n\n{clarification}"
            # Synthesize speech for clarification if voice replies enabled
            audio_response = None
            if voice_replies:
                show_processing_indicator(
                    status_placeholder,
                    "Preparing audio...",
                    "Converting to speech",
                    90
                )
                audio_response = synthesize_speech(client, clarification)
            
            status_placeholder.empty()
            st.session_state.pending_clarification = True
            st.session_state.pending_query = query
            st.session_state.clarification_question = clarification
            st.session_state.messages.append({"role": "assistant", "content": clarification, "audio": audio_response})
            with st.chat_message("assistant", avatar="🎬"):
                st.markdown(clarification)
                if audio_response:
                    st.audio(audio_response, format="audio/mp3", autoplay=auto_play_audio)
            return
        
        # Phase 2: Searching
        show_processing_indicator(
            status_placeholder,
            "Searching IMDB database...",
            "Scanning 1,000 top-rated films",
            40
        )
        
        _, results, recommendations, extras = run_query(
            df, embeddings, row_ids, client, query, conversation_history
        )
        
        # Phase 3: Generating response
        show_processing_indicator(
            status_placeholder,
            "Crafting your answer...",
            "Composing a helpful response",
            70
        )
        
        # Get year range from data for response context
        years = df["Released_Year"].dropna()
        data_min_year = int(years.min()) if len(years) > 0 else 1920
        data_max_year = int(years.max()) if len(years) > 0 else 2020
        
        response_text = build_response(
            client, query, results, recommendations, conversation_history,
            min_year=data_min_year, max_year=data_max_year, extras=extras
        )
        if tone_nudge:
            response_text = f"{tone_nudge}\n\n{response_text}"
        
        # Phase 4: Synthesizing speech
        show_processing_indicator(
            status_placeholder,
            "Preparing audio...",
            "Converting to speech",
            90
        )
        
        # Clean the response text for speech (remove URLs, markdown, etc.)
        speakable_text = clean_text_for_speech(response_text)
        audio_response = synthesize_speech(client, speakable_text) if voice_replies else None
        
        # Complete
        show_processing_indicator(
            status_placeholder,
            "Done!",
            "Ready to play",
            100
        )
        time.sleep(0.15)
        
        # Clear the processing indicator
        status_placeholder.empty()
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "audio": audio_response}
        )
        with st.chat_message("assistant", avatar="🎬"):
            st.markdown(response_text, unsafe_allow_html=True)
            if audio_response:
                st.audio(audio_response, format="audio/mp3", autoplay=auto_play_audio)

        # Render results
        render_results_section(results, recommendations)


if __name__ == "__main__":
    main()
