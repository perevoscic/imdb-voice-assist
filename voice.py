import os
import tempfile
from typing import Optional

from openai import OpenAI


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> Optional[str]:
    if not audio_bytes:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as handle:
            response = client.audio.transcriptions.create(
                model=os.getenv("STT_MODEL", "whisper-1"),
                file=handle,
            )
        return response.text
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def synthesize_speech(client: OpenAI, text: str) -> Optional[bytes]:
    if not text:
        return None
    response = client.audio.speech.create(
        model=os.getenv("TTS_MODEL", "tts-1"),
        voice=os.getenv("TTS_VOICE", "alloy"),
        input=text,
    )
    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    return response
