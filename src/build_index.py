import pickle
from typing import List

import faiss
import numpy as np
from openai import OpenAI

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)
from src.ingest import load_imdb


def _embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    embeddings: List[List[float]] = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in resp.data])
    return np.array(embeddings, dtype="float32")


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    df = load_imdb()
    if "Overview" not in df.columns:
        raise RuntimeError("Overview column missing in dataset")

    client = OpenAI(api_key=OPENAI_API_KEY)

    texts = df["Overview"].fillna("").astype(str).tolist()
    embeddings = _embed_texts(client, texts, OPENAI_EMBEDDING_MODEL)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = df.to_dict(orient="records")

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
