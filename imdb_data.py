import re
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class ImdbConfig:
    csv_path: str = "imdb_top_1000.csv"


NUMERIC_COLUMNS = [
    "Released_Year",
    "IMDB_Rating",
    "Meta_score",
    "No_of_votes",
    "Gross",
]

STAR_COLUMNS = ["Star1", "Star2", "Star3", "Star4"]


def _parse_int(value: str):
    if pd.isna(value):
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else None


def _parse_float(value: str):
    if pd.isna(value):
        return None
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else None


def load_imdb_data(config: ImdbConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path)

    if "No_of_Votes" in df.columns and "No_of_votes" not in df.columns:
        df = df.rename(columns={"No_of_Votes": "No_of_votes"})

    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df["Runtime_Min"] = (
        df["Runtime"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    df["No_of_votes"] = df["No_of_votes"].apply(_parse_int)
    df["Gross"] = df["Gross"].apply(_parse_float)

    df["Title_Lower"] = df["Series_Title"].astype(str).str.lower()
    df["Genre_Lower"] = df["Genre"].astype(str).str.lower()
    df["Director_Lower"] = df["Director"].astype(str).str.lower()
    df["Overview_Lower"] = df["Overview"].astype(str).str.lower()

    df["Stars_Lower"] = (
        df[STAR_COLUMNS]
        .astype(str)
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )

    df["Stars_List"] = df[STAR_COLUMNS].fillna("").values.tolist()
    df["Primary_Star"] = df["Star1"].fillna("")

    return df


def to_movie_cards(df: pd.DataFrame, limit: int) -> List[dict]:
    columns = [
        "Poster_Link",
        "Series_Title",
        "Released_Year",
        "Genre",
        "IMDB_Rating",
        "Meta_score",
        "Director",
        "Gross",
        "No_of_votes",
        "Overview",
    ]
    return df[columns].head(limit).to_dict(orient="records")
