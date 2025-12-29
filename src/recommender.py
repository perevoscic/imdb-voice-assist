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


def recommend_with_scores(
    semantic_engine: SemanticEngine,
    seed_records: List[Dict[str, Any]],
    k: int = 5,
    imdb_band: float = 0.5,
    meta_band: float = 10.0,
) -> List[Dict[str, Any]]:
    if not seed_records:
        return []

    seed = seed_records[0]
    overview = seed.get("Overview", "")
    if not overview:
        return []

    seed_imdb = seed.get("IMDB_Rating")
    seed_meta = seed.get("Meta_score")
    if seed_imdb is None or seed_meta is None:
        return recommend(semantic_engine, seed_records, k=k)

    allowed_indices = []
    for idx, record in enumerate(semantic_engine.metadata):
        imdb = record.get("IMDB_Rating")
        meta = record.get("Meta_score")
        if imdb is None or meta is None:
            continue
        if abs(imdb - seed_imdb) <= imdb_band and abs(meta - seed_meta) <= meta_band:
            allowed_indices.append(idx)

    if not allowed_indices:
        return recommend(semantic_engine, seed_records, k=k)

    results = semantic_engine.search_with_filter(overview, allowed_indices, k=k + 5)
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
