from typing import Dict, List, Optional

import pandas as pd


def build_response(
    client,
    query: str,
    results: List[dict],
    recommendations: List[dict],
    conversation_history: Optional[List[Dict]] = None,
    min_year: int = 1920,
    max_year: int = 2020,
    extras: Optional[Dict] = None,
) -> str:
    if extras and extras.get("compound_sections"):
        return _render_compound_sections(extras["compound_sections"])
    if extras and extras.get("summary_items"):
        return _render_summaries(
            extras["summary_items"],
            title=extras.get("summary_title", "Summaries"),
            note="Plot overviews are shortened for brevity.",
        )
    if extras and extras.get("actor_score_context"):
        return _render_actor_response(results, recommendations, extras["actor_score_context"])

    lowered = query.lower()
    if ("release" in lowered or "released" in lowered) and results:
        movie = results[0]
        year = movie.get("Released_Year")
        if year:
            if isinstance(year, float) and year.is_integer():
                year = int(year)
            return str(year)

    if not results:
        return "No matching movies were found for the given criteria."

    return _render_movie_cards(results)
    
    # LLM formatting removed to guarantee all results are shown.


def _render_movie_cards(results: List[dict]) -> str:
    def _fmt(value) -> str:
        if value is None or value == "" or pd.isna(value):
            return "Not available"
        return str(value)

    cards = []
    for movie in results:
        year = movie.get("Released_Year")
        if isinstance(year, float) and year.is_integer():
            year = int(year)
        card = (
            "<div style='display:flex;gap:16px;margin-bottom:20px;'>"
            f"<img src='{_fmt(movie.get('Poster_Link'))}' width='100'>"
            "<div>"
            f"<b>{_fmt(movie.get('Series_Title'))} ({_fmt(year)})</b><br>"
            f"Genre: {_fmt(movie.get('Genre'))}<br>"
            f"IMDb: {_fmt(movie.get('IMDB_Rating'))} | Meta: {_fmt(movie.get('Meta_score'))}<br>"
            f"Director: {_fmt(movie.get('Director'))}<br>"
            f"Overview: {_fmt(movie.get('Overview'))}"
            "</div>"
            "</div>"
        )
        cards.append(card)
    return "".join(cards)


def _render_actor_response(
    results: List[dict],
    recommendations: List[dict],
    context: Dict,
) -> str:
    def _fmt(value) -> str:
        if value is None or value == "" or pd.isna(value):
            return "Not available"
        return str(value)

    actor_name = context.get("actor") or "the actor"
    filters = context.get("filters", {})
    gross_min = filters.get("gross_min")
    imdb_min = filters.get("imdb_min")
    preference = context.get("preference") or "any"
    assumed_default = context.get("assumed_default", False)

    role_label = "as the lead actor" if preference == "lead" else "in any role"
    filter_bits = []
    if gross_min:
        filter_bits.append(f"gross over ${gross_min:,.0f}")
    if imdb_min:
        filter_bits.append(f"IMDb rating ≥ {imdb_min}")
    filter_text = f" filtered for {' and '.join(filter_bits)}" if filter_bits else ""

    preface = (
        f"Here are {actor_name} movies where they appear {role_label}{filter_text}."
        + (" You didn't specify a role preference, so I included any appearance." if assumed_default else "")
    )

    if not results:
        return (
            preface
            + " I couldn't find any matches. Try including supporting roles or lowering the revenue/rating thresholds."
        )

    parts: List[str] = [f"<p>{preface}</p>", _render_movie_cards(results)]

    if recommendations:
        rec_lines = [
            f"{_fmt(r.get('Series_Title'))} (IMDb {_fmt(r.get('IMDB_Rating'))} | Meta {_fmt(r.get('Meta_score'))})"
            for r in recommendations
        ]
        parts.append(
            "<div style='margin-top:12px;'>"
            "<b>Recommendations with similar IMDb/Meta scores:</b><br>"
            + "<br>".join(rec_lines)
            + "</div>"
        )

    parts.append(
        "<div style='color:#9ca3af;font-size:13px;margin-top:8px;'>"
        "Tell me if you prefer only lead roles, want supporting roles included, or would like different revenue/rating cut-offs."
        "</div>"
    )

    return "".join(parts)


def _render_compound_sections(sections: List[Dict]) -> str:
    def _fmt(value) -> str:
        if value is None or value == "" or pd.isna(value):
            return "Not available"
        return str(value)

    def _shorten(text: Optional[str], limit: int = 220) -> str:
        if not text:
            return "Not available"
        text = str(text).strip()
        if len(text) <= limit:
            return text
        cutoff = text.rfind(". ", 0, limit)
        if cutoff == -1:
            cutoff = limit
        return text[:cutoff].rstrip() + "..."

    parts: List[str] = []
    for section in sections:
        items = section.get("items") or []
        if not items:
            continue
        title = _fmt(section.get("title") or "")
        note = section.get("note")
        parts.append(
            "<div style='margin-bottom:22px;'>"
            f"<h4 style='margin:0 0 6px 0;'>{title}</h4>"
        )
        if note:
            parts.append(
                f"<div style='color:#9ca3af;font-size:13px;margin-bottom:8px;'>{_fmt(note)}</div>"
            )
        for movie in items:
            year = movie.get("Released_Year")
            if isinstance(year, float) and year.is_integer():
                year = int(year)
            overview = _shorten(movie.get("Overview"))
            card = (
                "<div style='display:flex;gap:14px;margin-bottom:14px;'>"
                f"<img src='{_fmt(movie.get('Poster_Link'))}' width='90'>"
                "<div>"
                f"<b>{_fmt(movie.get('Series_Title'))} ({_fmt(year)})</b><br>"
                f"IMDb: {_fmt(movie.get('IMDB_Rating'))} | Meta: {_fmt(movie.get('Meta_score'))}<br>"
                f"Genre: {_fmt(movie.get('Genre'))} | Director: {_fmt(movie.get('Director'))}<br>"
                f"Plot: {overview}"
                "</div>"
                "</div>"
            )
            parts.append(card)
        parts.append("</div>")
    return "".join(parts)


def _render_summaries(items: List[dict], title: str, note: Optional[str] = None) -> str:
    def _fmt(value) -> str:
        if value is None or value == "" or pd.isna(value):
            return "Not available"
        return str(value)

    def _shorten(text: Optional[str], limit: int = 200) -> str:
        if not text:
            return "Not available"
        text = str(text).strip()
        if len(text) <= limit:
            return text
        cutoff = text.rfind(". ", 0, limit)
        if cutoff == -1:
            cutoff = limit
        return text[:cutoff].rstrip() + "..."

    parts: List[str] = [
        "<div style='margin-bottom:20px;'>"
        f"<h4 style='margin:0 0 6px 0;'>{_fmt(title)}</h4>"
    ]
    if note:
        parts.append(
            f"<div style='color:#9ca3af;font-size:13px;margin-bottom:12px;'>{_fmt(note)}</div>"
        )
    
    # Grid container
    parts.append("<div style='display:grid; grid-template-columns: 1fr 1fr; gap: 20px;'>")

    for movie in items:
        year = movie.get("Released_Year")
        if isinstance(year, float) and year.is_integer():
            year = int(year)
        summary = _shorten(movie.get("Overview"), limit=200)
        
        poster_url = _fmt(movie.get('Poster_Link'))
        
        card = (
            "<div style='display:flex;gap:14px;align-items:start;background:var(--bg-tertiary);padding:16px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.05);'>"
            f"<img src='{poster_url}' width='70' style='border-radius:6px;object-fit:cover;aspect-ratio:2/3;flex-shrink:0;'>"
            "<div>"
            f"<strong style='display:block;margin-bottom:4px;font-size:1.05em;'>{_fmt(movie.get('Series_Title'))} ({_fmt(year)})</strong>"
            f"<div style='font-size:0.92em;opacity:0.9;line-height:1.5;'>{summary}</div>"
            "</div>"
            "</div>"
        )
        parts.append(card)
        
    parts.append("</div>") # Close grid
    parts.append("</div>") # Close outer wrapper
    return "".join(parts)
