from typing import Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_STT_MODEL, OPENAI_TTS_MODEL


class SpeechClient:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        if not audio_bytes:
            return ""
        resp = self.client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=(filename, audio_bytes),
        )
        return resp.text

    def synthesize(self, text: str, voice: str = "alloy") -> Optional[bytes]:
        if not text:
            return None
        resp = self.client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=text,
        )
        return resp.read()
