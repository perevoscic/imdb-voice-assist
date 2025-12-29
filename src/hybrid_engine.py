from typing import Any, Dict, List, Tuple

import pandas as pd

from src.semantic_engine import SemanticEngine
from src.structured_engine import StructuredQuery, run_structured_query


def hybrid_search(
    df: pd.DataFrame,
    semantic_engine: SemanticEngine,
    structured_query: StructuredQuery,
    semantic_query: str,
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    structured_df, reasoning = run_structured_query(df, structured_query)
    allowed_indices = structured_df.index.tolist()

    if allowed_indices:
        semantic_results = semantic_engine.search_with_filter(
            semantic_query, allowed_indices, k=k
        )
    else:
        semantic_results = semantic_engine.search(semantic_query, k=k)

    reasoning.append("semantic similarity over Overview")

    results = []
    for record, score, idx in semantic_results:
        record = dict(record)
        record["similarity"] = score
        record["_row_index"] = idx
        results.append(record)
    return results, reasoning
