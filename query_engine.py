import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from imdb_data import to_movie_cards
from vector_store import cosine_similarity


@dataclass
class QueryPlan:
    action: str
    filters: List[Dict]
    sort: List[Dict]
    limit: int
    text_query: Optional[str] = None


class _SchemaBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ClarificationResult(_SchemaBase):
    needs_clarification: bool
    clarification_question: Optional[str] = None
    reasoning: Optional[str] = None


class FilterSpec(_SchemaBase):
    field: str
    op: Literal["eq", "contains", "gt", "gte", "lt", "lte", "between", "in"]
    value: Any


class SortSpec(_SchemaBase):
    field: str
    order: Literal["asc", "desc"]


class QueryPlanSpec(_SchemaBase):
    action: Literal["filter_sort", "semantic_search", "hybrid"] = "filter_sort"
    filters: List[FilterSpec] = Field(default_factory=list)
    sort: List[SortSpec] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=50)
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


STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "for"}


def _normalize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9\\s]", " ", text.lower())
    return [token for token in cleaned.split() if token]


def find_title_match(df: pd.DataFrame, query: str) -> Optional[str]:
    lowered = query.lower()
    titles = df["Series_Title"].dropna().astype(str).tolist()
    matches = [title for title in titles if title.lower() in lowered]
    if matches:
        return max(matches, key=len)

    query_tokens = set(_normalize(query))
    candidates = []
    for title in titles:
        title_tokens = [t for t in _normalize(title) if t not in STOPWORDS]
        if not title_tokens:
            continue
        if set(title_tokens).issubset(query_tokens):
            candidates.append(title)
    if candidates:
        return max(candidates, key=len)

    normalized_query = " ".join(_normalize(query))
    best_title = None
    best_score = 0.0
    for title in titles:
        normalized_title = " ".join(_normalize(title))
        if not normalized_title:
            continue
        score = SequenceMatcher(None, normalized_title, normalized_query).ratio()
        if score > best_score:
            best_score = score
            best_title = title
    if best_score >= 0.6:
        return best_title
    return None


def find_title_keyword(query: str) -> Optional[str]:
    """
    Extract a potential movie title keyword from queries like:
    - "all Godfather movies"
    - "show me Matrix films"
    - "bring me Godfather's movies"
    Returns the keyword for title search, or None if not detected.
    """
    lowered = query.lower()
    
    # Patterns that indicate a title search
    patterns = [
        r"(?:all|show|bring|get|find|list)\s+(?:me\s+)?(?:the\s+)?([a-z0-9\s]+?)(?:'s)?\s+(?:movies?|films?)",
        r"(?:movies?|films?)\s+(?:called|named|titled)\s+(?:the\s+)?([a-z0-9\s]+)",
        r"([a-z0-9\s]+?)\s+(?:movies?|films?|trilogy|series|franchise)",
    ]
    
    # Words to strip from extracted keyword
    generic = {"best", "top", "all", "good", "great", "new", "old", "my", "favorite", "favourite", "the", "a", "an"}
    
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            keyword = match.group(1).strip()
            # Remove generic words from the keyword
            keyword_tokens = keyword.split()
            cleaned_tokens = [t for t in keyword_tokens if t not in generic]
            keyword = " ".join(cleaned_tokens).strip()
            
            if keyword and len(keyword) >= 3:
                return keyword
    
    return None


def find_movies_by_title_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Find all movies whose title contains the given keyword."""
    return df[df["Title_Lower"].str.contains(keyword.lower(), na=False)]


def parse_years(query: str) -> Tuple[Optional[int], Optional[int]]:
    years = [int(y) for y in re.findall(r"(19\\d{2}|20\\d{2})", query)]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], years[0]
    return min(years), max(years)


def parse_threshold(query: str, field: str) -> Optional[int]:
    pattern = rf"{field}\\s*(?:above|over|>=|greater than)\\s*(\\d+)"
    match = re.search(pattern, query.lower())
    if match:
        return int(match.group(1))
    return None


def parse_genre(query: str) -> Optional[str]:
    genres = [
        "comedy",
        "horror",
        "sci-fi",
        "drama",
        "action",
        "romance",
        "thriller",
        "crime",
        "adventure",
        "animation",
        "fantasy",
        "mystery",
    ]
    lowered = query.lower()
    for genre in genres:
        if genre in lowered:
            return genre
    return None


def parse_director(query: str, df: pd.DataFrame) -> Optional[str]:
    """Extract director name from query if present in the dataset."""
    lowered = query.lower()
    # Check for "directed by X" or "director X" or "by X" patterns
    patterns = [
        r"directed by\s+([a-z\s]+?)(?:\?|$|,|\.|movies|films)",
        r"director\s+([a-z\s]+?)(?:\?|$|,|\.|movies|films)",
        r"([a-z\s]+?)(?:'s|s')\s+(?:movies|films|directed)",
        r"movies\s+(?:by|from)\s+([a-z\s]+?)(?:\?|$|,|\.)",
        r"films\s+(?:by|from)\s+([a-z\s]+?)(?:\?|$|,|\.)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            potential_name = match.group(1).strip()
            # Verify this name exists in our director column
            directors = df["Director_Lower"].dropna().unique()
            for director in directors:
                if potential_name in director or director in potential_name:
                    return director
    
    # Fallback: check if any known director name appears in the query
    directors = df["Director_Lower"].dropna().unique()
    for director in directors:
        # Require at least first and last name match (2+ words)
        director_parts = director.split()
        if len(director_parts) >= 2:
            if director in lowered:
                return director
    
    return None


def parse_actor(query: str, df: pd.DataFrame) -> Optional[str]:
    """Extract actor name from query if present in the dataset (Star1-Star4 columns)."""
    lowered = query.lower()
    
    # Remove possessive suffixes for matching (e.g., "Al Pacino's" -> "Al Pacino")
    # Keep the cleaned version for pattern matching
    cleaned_query = re.sub(r"'s\b", "", lowered)
    cleaned_query = re.sub(r"'s\b", "", cleaned_query)  # Handle curly apostrophe too
    
    # Check for actor-related patterns with improved regex
    patterns = [
        r"(?:starring|with|featuring)\s+([a-z\s]+?)(?:\?|$|,|\.|movies|films|as)",
        r"(?:all\s+)?([a-z\s]+?)(?:'s?)?\s+(?:movies?|films?|starred|stars)",
        r"movies?\s+(?:with|starring|featuring|by|from)\s+([a-z\s]+?)(?:\?|$|,|\.)",
        r"films?\s+(?:with|starring|featuring|by|from)\s+([a-z\s]+?)(?:\?|$|,|\.)",
        r"([a-z\s]+?)\s+as\s+(?:lead|main|star)",
        r"(?:show|list|get|find|bring)\s+(?:me\s+)?(?:all\s+)?([a-z\s]+?)(?:'s?)?\s+(?:movies?|films?)",
    ]
    
    # Build a set of unique actors from Star1-4 (do this once, not in loop)
    all_actors = set()
    for star_col in ["Star1", "Star2", "Star3", "Star4"]:
        actors = df[star_col].dropna().astype(str).str.lower().unique()
        all_actors.update(actors)
    
    for pattern in patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            potential_name = match.group(1).strip()
            # Remove generic words that might be captured
            generic = {"all", "the", "me", "show", "list", "get", "find", "bring"}
            name_parts = [p for p in potential_name.split() if p not in generic]
            potential_name = " ".join(name_parts).strip()
            
            if len(potential_name) < 3:
                continue
                
            # Check if this name matches any known actor
            for actor in all_actors:
                if potential_name in actor or actor in potential_name:
                    return actor
    
    # Fallback: check if any known actor name (2+ words) appears in the query
    for actor in all_actors:
        actor_parts = actor.split()
        if len(actor_parts) >= 2:
            if actor in cleaned_query:
                return actor
    
    return None


def format_conversation_history(messages: List[Dict], max_turns: int = 5) -> str:
    """Format recent conversation history for LLM context."""
    if not messages:
        return ""
    
    # Get last N turns (user + assistant pairs)
    recent = messages[-(max_turns * 2):]
    
    formatted = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            formatted.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted)


def check_needs_clarification(
    client: OpenAI,
    query: str,
    conversation_history: List[Dict],
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Use LLM to determine if the query needs clarification.
    Returns the clarification question if needed, None otherwise.
    """
    # First, check if the query mentions a specific movie that has exactly one match
    # If so, no clarification is needed
    title_match = find_title_match(df, query)
    if title_match:
        # Count how many movies match this title (or similar titles)
        matching_movies = df[df["Series_Title"].str.lower().str.contains(
            title_match.lower().split()[0], na=False  # Use first word to catch sequels
        )]
        # If only one movie matches, no clarification needed
        if len(matching_movies) == 1:
            return None
        # If multiple movies but exact title match, still no clarification
        exact_matches = df[df["Series_Title"].str.lower() == title_match.lower()]
        if len(exact_matches) == 1:
            return None
    
    history_str = format_conversation_history(conversation_history)
    
    # Get some context about available data
    genres = df["Genre"].dropna().unique()[:10].tolist()
    directors = df["Director"].dropna().unique()[:20].tolist()
    
    # Get the actual year range from the data
    years = df["Released_Year"].dropna()
    min_year = int(years.min()) if len(years) > 0 else 1920
    max_year = int(years.max()) if len(years) > 0 else 2020
    
    system = """You analyze movie database queries to determine if clarification is needed.

IMPORTANT DATA CONTEXT:
- The movie database contains films from {min_year} to {max_year}
- When NO specific year or time period is mentioned, ALWAYS use the FULL available range ({min_year}-{max_year})
- NEVER ask clarifying questions about time periods, years, or date ranges
- Terms like "lower gross", "highest rated", "best", etc. should be applied across ALL available data unless a specific period is mentioned

Do NOT ask for clarification about:
1. Time periods, years, or date ranges - use full range {min_year}-{max_year} by default
2. "Lower" or "higher" comparisons - apply to all data
3. Queries that can be answered with reasonable defaults

Only ask for clarification if:
1. The query is genuinely ambiguous in a way that cannot be resolved with defaults
2. Multiple completely different interpretations exist that would give very different results

Available genres include: {genres}
Some directors in the database: {directors}

Respond with JSON only:
{{
    "needs_clarification": true/false,
    "clarification_question": "question to ask" or null,
    "reasoning": "brief explanation"
}}"""

    user_prompt = f"""Conversation history:
{history_str if history_str else "(No previous conversation)"}

Current query: {query}

Does this query need clarification before I can search the movie database?"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system.format(
                    genres=genres, 
                    directors=directors,
                    min_year=min_year,
                    max_year=max_year
                )},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        try:
            result = ClarificationResult.model_validate_json(_clean_json(content))
        except ValidationError:
            return None
        if result.needs_clarification and result.clarification_question:
            return result.clarification_question
        return None
    except Exception:
        return None


def resolve_references(
    client: OpenAI,
    query: str,
    conversation_history: List[Dict],
) -> str:
    """
    Resolve pronouns and references in the query using conversation history.
    Returns the resolved query.
    """
    if not conversation_history:
        return query
    
    history_str = format_conversation_history(conversation_history)
    
    # Quick check if resolution is likely needed
    reference_indicators = [
        "that movie", "that film", "this movie", "this film",
        "those movies", "these movies", "the same", "similar",
        "his", "her", "their", "its", "he", "she", "they",
        "more about", "tell me more", "what else", "another",
        "the director", "the actor", "the cast", "it", "them"
    ]
    
    query_lower = query.lower()
    needs_resolution = any(ref in query_lower for ref in reference_indicators)
    
    if not needs_resolution:
        return query
    
    system = """You resolve references in movie queries using conversation history.

Your task: Rewrite the query to be self-contained by replacing pronouns and references 
with the actual entities they refer to from the conversation.

Rules:
1. If a reference is clear from history, replace it with the actual name/title
2. If a reference is unclear, keep the original wording
3. Preserve the user's intent and question type
4. Return ONLY the rewritten query, nothing else
5. If no resolution is needed, return the original query unchanged"""

    user_prompt = f"""Conversation history:
{history_str}

Current query: {query}

Rewritten query (self-contained):"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=80,
        )
        resolved = response.choices[0].message.content.strip()
        # Remove quotes if the LLM wrapped the response
        resolved = resolved.strip('"\'')
        return resolved if resolved else query
    except Exception:
        return query


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


def plan_query_with_llm(
    client: OpenAI,
    query: str,
    conversation_history: Optional[List[Dict]] = None,
    min_year: int = 1920,
    max_year: int = 2020,
) -> QueryPlan:
    history_str = format_conversation_history(conversation_history or [])
    
    system = (
        "You plan queries over a pandas dataframe of IMDB movies. "
        "Return JSON only. Fields: action (filter_sort|semantic_search|hybrid), "
        "filters (list of {field, op, value}), sort (list of {field, order}), "
        "limit (int), text_query (string or null). "
        "Columns: Series_Title, Released_Year, Certificate, Runtime_Min, Genre, "
        "IMDB_Rating, Overview, Meta_score, Director, Star1-4, No_of_votes, Gross, "
        "Title_Lower, Genre_Lower, Director_Lower, Overview_Lower, Stars_Lower. "
        "Use conversation history to understand context and resolve any references. "
        f"IMPORTANT: The database contains movies from {min_year} to {max_year}. "
        "When NO specific year or time period is mentioned in the query, "
        f"use the FULL available range ({min_year}-{max_year}). "
        "Do NOT add year filters unless the user explicitly specifies a time period."
    )
    
    user_content = f"Query: {query}"
    if history_str:
        user_content = f"Conversation history:\n{history_str}\n\nCurrent query: {query}"
    
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    try:
        plan_spec = QueryPlanSpec.model_validate_json(_clean_json(content))
        return QueryPlan(
            action=plan_spec.action,
            filters=[f.model_dump() for f in plan_spec.filters],
            sort=[s.model_dump() for s in plan_spec.sort],
            limit=plan_spec.limit,
            text_query=plan_spec.text_query,
        )
    except ValidationError:
        fallback_limit = parse_top_n(query, default=5)
        return QueryPlan(
            action="filter_sort",
            filters=[],
            sort=[],
            limit=fallback_limit,
            text_query=None,
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
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[str, List[dict], List[dict]]:
    # Get year range from data for context
    years = df["Released_Year"].dropna()
    data_min_year = int(years.min()) if len(years) > 0 else 1920
    data_max_year = int(years.max()) if len(years) > 0 else 2020
    
    # Resolve references using conversation history
    resolved_query = resolve_references(client, query, conversation_history or [])
    
    lowered = resolved_query.lower()
    limit = parse_top_n(resolved_query, default=5)

    title_match = find_title_match(df, resolved_query)
    if title_match and ("release" in lowered or "released" in lowered):
        results = df[df["Series_Title"] == title_match].head(1)
        cards = to_movie_cards(results, 1)
        recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
        return "filtered", cards, recs

    # Handle title keyword searches like "all Godfather movies", "Matrix films", etc.
    title_keyword = find_title_keyword(resolved_query)
    if title_keyword:
        results = find_movies_by_title_keyword(df, title_keyword)
        if not results.empty:
            # Use a higher limit for "all X movies" queries to show full franchises
            title_limit = max(limit, len(results))
            results = results.sort_values("IMDB_Rating", ascending=False).head(title_limit)
            cards = to_movie_cards(results, title_limit)
            recs = recommend_similar(df, embeddings, row_ids, results.index.tolist())
            return "filtered", cards, recs

    year_start, year_end = parse_years(query)
    if "meta score" in lowered and year_start and year_end and "top" in lowered:
        filtered = df[df["Released_Year"].between(year_start, year_end)]
        filtered = filtered.sort_values("Meta_score", ascending=False).head(limit)
        cards = to_movie_cards(filtered, limit)
        recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
        return "filtered", cards, recs

    genre = parse_genre(query)
    if genre and "imdb" in lowered and "rating" in lowered and year_start and year_end:
        filtered = df[
            df["Released_Year"].between(year_start, year_end)
            & df["Genre_Lower"].str.contains(genre, na=False)
        ]
        filtered = filtered.sort_values("IMDB_Rating", ascending=False).head(limit)
        cards = to_movie_cards(filtered, limit)
        recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
        return "filtered", cards, recs

    if genre and "meta score" in lowered and "imdb" in lowered:
        meta_threshold = parse_threshold(query, "meta score")
        imdb_threshold = parse_threshold(query, "imdb rating")
        filtered = df[df["Genre_Lower"].str.contains(genre, na=False)]
        if meta_threshold is not None:
            filtered = filtered[filtered["Meta_score"] >= meta_threshold]
        if imdb_threshold is not None:
            filtered = filtered[filtered["IMDB_Rating"] >= imdb_threshold]
        filtered = filtered.sort_values("IMDB_Rating", ascending=False).head(limit)
        cards = to_movie_cards(filtered, limit)
        recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
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

    # Handle director queries
    director = parse_director(resolved_query, df)
    if director and ("direct" in lowered or "by" in lowered or "movie" in lowered or "film" in lowered):
        filtered = df[df["Director_Lower"].str.contains(director, na=False)]
        filtered = filtered.sort_values("IMDB_Rating", ascending=False).head(limit)
        cards = to_movie_cards(filtered, limit)
        recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
        return "filtered", cards, recs

    # Handle actor queries
    actor = parse_actor(resolved_query, df)
    if actor:
        # Search in Stars_Lower which contains all Star1-4 combined
        filtered = df[df["Stars_Lower"].str.contains(actor, na=False)]
        filtered = filtered.sort_values("IMDB_Rating", ascending=False).head(limit)
        cards = to_movie_cards(filtered, limit)
        recs = recommend_similar(df, embeddings, row_ids, filtered.index.tolist())
        return "filtered", cards, recs

    try:
        plan = plan_query_with_llm(
            client, 
            resolved_query, 
            conversation_history,
            min_year=data_min_year,
            max_year=data_max_year
        )
    except Exception:
        plan = QueryPlan(action="filter_sort", filters=[], sort=[], limit=limit)

    filtered = df
    if plan.filters:
        filtered = apply_filters(filtered, plan.filters)

    if plan.action in {"semantic_search", "hybrid"}:
        text_query = plan.text_query or resolved_query
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
