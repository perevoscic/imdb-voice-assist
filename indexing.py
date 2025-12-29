from typing import List, Tuple

import numpy as np
from openai import OpenAI

from vector_store import normalize_embeddings


def batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def build_embeddings(
    client: OpenAI,
    texts: List[str],
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    embeddings_list = []
    for batch in batched(texts, 64):
        response = client.embeddings.create(model=model, input=batch)
        embeddings_list.extend([item.embedding for item in response.data])
    embeddings = np.array(embeddings_list, dtype=np.float32)
    return normalize_embeddings(embeddings)
