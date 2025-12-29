from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class StructuredQuery:
    filters: Dict[str, Any]
    sort: Optional[Dict[str, str]] = None
    limit: Optional[int] = None
    groupby: Optional[Dict[str, Any]] = None


def _genre_match(genres: List[str], target: str) -> bool:
    target_lower = target.lower()
    return any(target_lower in g.lower() for g in genres)


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    reasoning = []
    out = df

    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min is not None:
        out = out[out["Released_Year"] >= year_min]
        reasoning.append(f"year >= {year_min}")
    if year_max is not None:
        out = out[out["Released_Year"] <= year_max]
        reasoning.append(f"year <= {year_max}")

    genre = filters.get("genre")
    if genre:
        out = out[out["Genre_list"].apply(lambda gs: _genre_match(gs, genre))]
        reasoning.append(f"genre contains {genre}")

    imdb_min = filters.get("imdb_min")
    if imdb_min is not None:
        out = out[out["IMDB_Rating"] >= imdb_min]
        reasoning.append(f"IMDB_Rating >= {imdb_min}")

    metascore_min = filters.get("metascore_min")
    if metascore_min is not None:
        out = out[out["Meta_score"] >= metascore_min]
        reasoning.append(f"Meta_score >= {metascore_min}")

    votes_min = filters.get("votes_min")
    if votes_min is not None:
        out = out[out["No_of_votes"] >= votes_min]
        reasoning.append(f"No_of_votes >= {votes_min}")

    gross_min = filters.get("gross_min")
    if gross_min is not None:
        out = out[out["Gross_num"] >= gross_min]
        reasoning.append(f"Gross_num >= {gross_min}")

    director = filters.get("director")
    if director:
        out = out[out["Director"].str.contains(director, case=False, na=False)]
        reasoning.append(f"Director contains {director}")

    title = filters.get("title")
    if title:
        out = out[out["Series_Title"].str.contains(title, case=False, na=False)]
        reasoning.append(f"Series_Title contains {title}")

    return out, reasoning


def apply_sort(df: pd.DataFrame, sort: Optional[Dict[str, str]]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not sort:
        return df, None
    column = sort.get("column")
    order = sort.get("order", "desc")
    if column not in df.columns:
        return df, None
    ascending = order == "asc"
    out = df.sort_values(by=column, ascending=ascending)
    return out, f"sorted by {column} {order}"


def apply_groupby(df: pd.DataFrame, groupby: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not groupby:
        return df, None
    column = groupby.get("column")
    if column not in df.columns:
        return df, None
    agg = groupby.get("agg", "count")
    if agg == "count":
        grouped = df.groupby(column).size().reset_index(name="count")
    else:
        metric = groupby.get("metric")
        if metric and metric in df.columns:
            grouped = df.groupby(column)[metric].agg(agg).reset_index()
        else:
            grouped = df.groupby(column).size().reset_index(name="count")
    return grouped, f"grouped by {column}"


def run_structured_query(df: pd.DataFrame, query: StructuredQuery) -> Tuple[pd.DataFrame, List[str]]:
    filtered, reasoning = apply_filters(df, query.filters)
    grouped, group_reason = apply_groupby(filtered, query.groupby)
    if group_reason:
        reasoning.append(group_reason)
    sorted_df, sort_reason = apply_sort(grouped, query.sort)
    if sort_reason:
        reasoning.append(sort_reason)
    if query.limit:
        sorted_df = sorted_df.head(query.limit)
        reasoning.append(f"limit {query.limit}")
    return sorted_df, reasoning
