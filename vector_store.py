import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class VectorStorePaths:
    embeddings_path: str = "data/imdb_embeddings.npz"
    meta_path: str = "data/imdb_embeddings_meta.json"


def load_vector_store(paths: VectorStorePaths) -> Tuple[np.ndarray, List[int]]:
    if not os.path.exists(paths.embeddings_path):
        return np.array([]), []
    data = np.load(paths.embeddings_path)
    embeddings = data["embeddings"]
    row_ids = data["row_ids"].tolist()
    return embeddings, row_ids


def save_vector_store(
    embeddings: np.ndarray,
    row_ids: List[int],
    paths: VectorStorePaths,
):
    os.makedirs(os.path.dirname(paths.embeddings_path), exist_ok=True)
    np.savez_compressed(paths.embeddings_path, embeddings=embeddings, row_ids=row_ids)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([])
    query_norm = query / (np.linalg.norm(query) or 1.0)
    return embeddings @ query_norm


def save_meta(meta: Dict, paths: VectorStorePaths):
    os.makedirs(os.path.dirname(paths.meta_path), exist_ok=True)
    with open(paths.meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)


def load_meta(paths: VectorStorePaths) -> Dict:
    if not os.path.exists(paths.meta_path):
        return {}
    with open(paths.meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
