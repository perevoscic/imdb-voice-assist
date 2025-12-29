import pickle
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from openai import OpenAI

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)


class SemanticEngine:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.metadata: List[Dict[str, Any]] = pickle.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[query])
        vec = np.array(resp.data[0].embedding, dtype="float32")[None, :]
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float, int]]:
        vec = self._embed_query(query)
        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score), int(idx)))
        return results

    def search_with_filter(
        self, query: str, allowed_indices: List[int], k: int = 5
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        vec = self._embed_query(query)
        scores, indices = self.index.search(vec, max(k * 5, 10))
        results = []
        allowed_set = set(allowed_indices)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx not in allowed_set:
                continue
            results.append((self.metadata[idx], float(score), int(idx)))
            if len(results) >= k:
                break
        return results
