from typing import Any, Dict, List

from src.semantic_engine import SemanticEngine


def recommend(
    semantic_engine: SemanticEngine,
    seed_records: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    if not seed_records:
        return []

    seed = seed_records[0]
    overview = seed.get("Overview", "")
    if not overview:
        return []

    results = semantic_engine.search(overview, k=k + 5)
    seed_titles = {rec.get("Series_Title") for rec in seed_records}

    recommendations = []
    for record, score, idx in results:
        if record.get("Series_Title") in seed_titles:
            continue
        record = dict(record)
        record["similarity"] = score
        record["_row_index"] = idx
        recommendations.append(record)
        if len(recommendations) >= k:
            break

    return recommendations
