import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "vectorstore"

load_dotenv(dotenv_path=ROOT / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")

DATA_PATH = DATA_DIR / "imdb.csv"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "imdb_faiss.index"
METADATA_PATH = ARTIFACTS_DIR / "imdb_metadata.pkl"
