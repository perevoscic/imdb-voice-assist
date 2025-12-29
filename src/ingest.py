import re
from typing import List

import numpy as np
import pandas as pd

from src.config import DATA_PATH


def _to_int(value):
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_year(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d{4}", str(value))
    return _to_int(match.group(0)) if match else np.nan


def _parse_runtime(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return _to_int(match.group(0)) if match else np.nan


def _parse_gross(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    return _to_int(cleaned)


def _parse_genres(value) -> List[str]:
    if pd.isna(value):
        return []
    return [genre.strip() for genre in str(value).split(",") if genre.strip()]


def load_imdb(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    df = df.rename(columns={"No_of_Votes": "No_of_votes"})

    required = [
        "Poster_Link",
        "Series_Title",
        "Released_Year",
        "Certificate",
        "Runtime",
        "Genre",
        "IMDB_Rating",
        "Overview",
        "Meta_score",
        "Director",
        "Star1",
        "Star2",
        "Star3",
        "Star4",
        "No_of_votes",
        "Gross",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "Released_Year" in df.columns:
        df["Released_Year"] = df["Released_Year"].apply(_parse_year)
    if "Runtime" in df.columns:
        df["Runtime_min"] = df["Runtime"].apply(_parse_runtime)
    if "Gross" in df.columns:
        df["Gross_num"] = df["Gross"].apply(_parse_gross)
    if "Genre" in df.columns:
        df["Genre_list"] = df["Genre"].apply(_parse_genres)
    if "IMDB_Rating" in df.columns:
        df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    if "Meta_score" in df.columns:
        df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    if "No_of_votes" in df.columns:
        df["No_of_votes"] = pd.to_numeric(df["No_of_votes"], errors="coerce")

    return df
