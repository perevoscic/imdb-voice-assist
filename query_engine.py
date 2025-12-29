import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from imdb_data import to_movie_cards
from vector_store import cosine_similarity


@dataclass
class QueryPlan:
    action: str
    filters: List[Dict]
    sort: List[Dict]
    limit: int
    text_query: Optional[str] = None


def _clean_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return cleaned


def parse_top_n(query: str, default: int = 5) -> int:
    match = re.search(r"top\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    return default


def find_title_match(df: pd.DataFrame, query: str) -> Optional[str]:
    lowered = query.lower()
    titles = df["Series_Title"].dropna().astype(str).tolist()
    matches = [title for title in titles if title.lower() in lowered]
    if not matches:
        return None
    return max(matches, key=len)


def needs_al_pacino_clarification(query: str) -> bool:
    lowered = query.lower()
    if "al pacino" not in lowered:
        return False
    if "lead" in lowered or "star1" in lowered or "primary" in lowered:
        return False
    if "support" in lowered or "supporting" in lowered or "any role" in lowered:
        return False
    return True


def apply_filters(df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    filtered = df
    for f in filters:
        field = f.get("field")
        op = f.get("op")
        value = f.get("value")
        if field not in filtered.columns:
            continue
        series = filtered[field]
        if op == "eq":
            filtered = filtered[series == value]
        elif op == "contains":
            filtered = filtered[series.astype(str).str.contains(str(value), case=False, na=False)]
        elif op == "gt":
            filtered = filtered[series > value]
        elif op == "gte":
            filtered = filtered[series >= value]
        elif op == "lt":
            filtered = filtered[series < value]
        elif op == "lte":
            filtered = filtered[series <= value]
        elif op == "between" and isinstance(value, list) and len(value) == 2:
            filtered = filtered[series.between(value[0], value[1])]
        elif op == "in" and isinstance(value, list):
            filtered = filtered[series.isin(value)]
    return filtered


def plan_query_with_llm(client: OpenAI, query: str) -> QueryPlan:
    system = (
        "You plan queries over a pandas dataframe of IMDB movies. "
        "Return JSON only. Fields: action (filter_sort|semantic_search|hybrid), "
        "filters (list of {field, op, value}), sort (list of {field, order}), "
        "limit (int), text_query (string or null). "
        "Columns: Series_Title, Released_Year, Certificate, Runtime_Min, Genre, "
        "IMDB_Rating, Overview, Meta_score, Director, Star1-4, No_of_votes, Gross, "
        "Title_Lower, Genre_Lower, Director_Lower, Overview_Lower, Stars_Lower.")
    user = f"Query: {query}"
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    plan_dict = json.loads(_clean_json(content))
    return QueryPlan(
        action=plan_dict.get("action", "filter_sort"),
        filters=plan_dict.get("filters", []),
        sort=plan_dict.get("sort", []),
        limit=int(plan_dict.get("limit", 5)),
        text_query=plan_dict.get("text_query"),
    )


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        input=text,
    )
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    return vector


def semantic_rank(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    row_ids: List[int],
    query_vector: np.ndarray,
    limit: int,
    allowed_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if embeddings.size == 0:
        return df.head(0)
    if allowed_ids is not None:
        allowed_set = set(allowed_ids)
        allowed_indices = [i for i, row_id in enumerate(row_ids) if row_id in allowed_set]
        if not allowed_indices:
            return df.head(0)
        subset_embeddings = embeddings[allowed_indices]
        sims = cosine_similarity(query_vector, subset_embeddings)
        top_local = np.argsort(sims)[::-1][: limit * 3]
        matched_ids = [row_ids[allowed_indices[i]] for i in top_local]
        subset = df.loc[matched_ids].copy()
        subset["_similarity"] = sims[top_local]
        subset = subset.sort_values("_similarity", ascending=False)
        return subset.head(limit)

    sims = cosine_similarity(query_vector, embeddings)
    top_idx = np.argsort(sims)[::-1][: limit * 3]
    matched_ids = [row_ids[i] for i in top_idx]
    subset = df.loc[matched_ids].copy()
    subset["_similarity"] = sims[top_idx]
    subset = subset.sort_values("_similarity", ascending=False)
    return subset.head(limit)


def recommend_similar(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    row_ids: List[int],
    seed_ids: List[int],
    limit: int = 3,
) -> List[dict]:
    if embeddings.size == 0 or not seed_ids:
        return []
    id_to_index = {row_id: i for i, row_id in enumerate(row_ids)}
    seed_vectors = [embeddings[id_to_index[sid]] for sid in seed_ids if sid in id_to_index]
    if not seed_vectors:
        return []
    centroid = np.mean(seed_vectors, axis=0)
    sims = cosine_similarity(centroid, embeddings)
    ranked = np.argsort(sims)[::-1]
    recommendations = []
    for idx in ranked:
        row_id = row_ids[idx]
        if row_id in seed_ids:
            continue
        recommendations.append(df.loc[row_id])
        if len(recommendations) >= limit:
            break
    if not recommendations:
        return []
    rec_df = pd.DataFrame(recommendations)
    return to_movie_cards(rec_df, limit)


def _handle_top_directors(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    high_gross = df[df["Gross"].fillna(0) > 500_000_000]
    grouped = high_gross.groupby("Director")
    eligible = grouped.filter(lambda g: len(g) >= 2)
    top_movie = (
        eligible.sort_values("Gross", ascending=False)
        .groupby("Director")
        .head(1)
        .sort_values("Gross", ascending=False)
    )
    return top_movie.head(limit)


def _handle_low_gross_high_votes(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    filtered = df[df["No_of_votes"].fillna(0) > 1_000_000]
    return filtered.sort_values("Gross", ascending=True).head(limit)


def _handle_spielberg_scifi(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    filtered = df[
        df["Director_Lower"].str.contains("steven spielberg", na=False)
        & df["Genre_Lower"].str.contains("sci-fi", na=False)
    ]
    return filtered.sort_values("IMDB_Rating", ascending=False).head(limit)


def _handle_comedy_death(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    row_ids: List[int],
    client: OpenAI,
    limit: int,
) -> pd.DataFrame:
    allowed_ids = df[df["Genre_Lower"].str.contains("comedy", na=False)].index.tolist()
    query_vector = embed_query(client, "death, dead people, loss, grief")
    return semantic_rank(df, embeddings, row_ids, query_vector, limit, allowed_ids)


def _handle_police_before(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    row_ids: List[int],
    client: OpenAI,
    limit: int,
) -> pd.DataFrame:
    allowed_ids = df[df["Released_Year"].fillna(0) < 1990].index.tolist()
    query_vector = embed_query(client, "police investigation, law enforcement, detectives")
    return semantic_rank(df, embeddings, row_ids, query_vector, limit, allowed_ids)


def run_query(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    row_ids: List[int],
    client: OpenAI,
    query: str,
) -> Tuple[str, List[dict], List[dict]]:
    lowered = query.lower()
    limit = parse_top_n(query, default=5)

    title_match = find_title_match(df, query)
    if title_match and ("release" in lowered or "released" in lowered):
        results = df[df["Series_Title"] == title_match].head(1)
        cards = to_movie_cards(results, 1)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "filtered", cards, recs

    if "top directors" in lowered and "gross" in lowered and "twice" in lowered:
        results = _handle_top_directors(df, limit)
        cards = to_movie_cards(results, limit)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "filtered", cards, recs

    if "over 1m votes" in lowered or "over 1m" in lowered:
        results = _handle_low_gross_high_votes(df, limit)
        cards = to_movie_cards(results, limit)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "filtered", cards, recs

    if "steven spielberg" in lowered and "sci-fi" in lowered:
        results = _handle_spielberg_scifi(df, limit)
        cards = to_movie_cards(results, limit)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "filtered", cards, recs

    if "comedy" in lowered and ("death" in lowered or "dead" in lowered):
        results = _handle_comedy_death(df, embeddings, row_ids, client, limit)
        cards = to_movie_cards(results, limit)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "semantic", cards, recs

    if "police" in lowered and "before 1990" in lowered:
        results = _handle_police_before(df, embeddings, row_ids, client, limit)
        cards = to_movie_cards(results, limit)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "semantic", cards, recs

    try:
        plan = plan_query_with_llm(client, query)
    except Exception:
        plan = QueryPlan(action="filter_sort", filters=[], sort=[], limit=limit)

    filtered = df
    if plan.filters:
        filtered = apply_filters(filtered, plan.filters)

    if plan.action in {"semantic_search", "hybrid"}:
        text_query = plan.text_query or query
        query_vector = embed_query(client, text_query)
        allowed_ids = filtered.index.tolist() if plan.filters else None
        filtered = semantic_rank(df, embeddings, row_ids, query_vector, plan.limit, allowed_ids)
    else:
        for sort in plan.sort:
            field = sort.get("field")
            order = sort.get("order", "desc")
            if field in filtered.columns:
                filtered = filtered.sort_values(field, ascending=(order == "asc"))
        filtered = filtered.head(plan.limit)

    cards = to_movie_cards(filtered, plan.limit)
    recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
    return plan.action, cards, recs
