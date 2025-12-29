import os
import re
import tempfile
from typing import Optional

from openai import OpenAI


def clean_text_for_speech(text: str) -> str:
    """Remove URLs, markdown, and other non-speakable content from text."""
    if not text:
        return ""
    
    # Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove image markdown ![alt](url) and [text](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove "Poster:" or "Poster_Link:" labels and their values
    text = re.sub(r'\*?\*?Poster(?:_Link)?:?\*?\*?\s*\S*', '', text, flags=re.IGNORECASE)
    
    # Remove standalone URLs that might be on their own line
    text = re.sub(r'^\s*https?://.*$', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bullet points but keep the text
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
    
    # Remove numbered list markers but keep the text
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove empty parentheses or brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    return text.strip()


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
