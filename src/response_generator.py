from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


class ResponseGenerator:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)

    def generate(
        self,
        user_text: str,
        results: List[Dict[str, Any]],
        reasoning: List[str],
        recommendations: Optional[List[Dict[str, Any]]] = None,
        needs_plot_summary: bool = False,
    ) -> str:
        user_lower = user_text.lower()
        if (
            "when did" in user_lower
            and "release" in user_lower
            and results
            and results[0].get("Released_Year") is not None
        ):
            title = results[0].get("Series_Title", "That title")
            year = results[0].get("Released_Year")
            try:
                year_int = int(year)
                return f"{title} was released in {year_int}."
            except (TypeError, ValueError):
                return f"I couldn't find the release year for {title}."
        summary_lines = []
        overview_lines = []
        for idx, item in enumerate(results[:8]):
            title = item.get("Series_Title", "Unknown")
            year = item.get("Released_Year", "")
            try:
                year_display = str(int(year)) if year != "" else ""
            except (TypeError, ValueError):
                year_display = ""
            rating = item.get("IMDB_Rating", "")
            meta = item.get("Meta_score", "")
            summary_lines.append(
                f"- {title} ({year_display}) | IMDB {rating} | Meta {meta}"
            )
            if needs_plot_summary and idx < 5:
                overview = item.get("Overview", "")
                if overview:
                    overview_lines.append(f"- {title}: {overview}")

        rec_lines = []
        if recommendations:
            for item in recommendations[:5]:
                title = item.get("Series_Title", "Unknown")
                year = item.get("Released_Year", "")
                try:
                    year_display = str(int(year)) if year != "" else ""
                except (TypeError, ValueError):
                    year_display = ""
                rating = item.get("IMDB_Rating", "")
                meta = item.get("Meta_score", "")
                rec_lines.append(
                    f"- {title} ({year_display}) | IMDB {rating} | Meta {meta}"
                )

        prompt = """
You are a movie assistant. Provide a concise conversational answer.
Include a short reasoning sentence based on the provided reasoning list.
If recommendations are provided, add a "Similar movies" section.
If plot summaries are requested, summarize each movie in 1-2 sentences.
""".strip()

        content = [
            f"User question: {user_text}",
            "Results:",
            "\n".join(summary_lines) if summary_lines else "(no results)",
            f"Reasoning signals: {', '.join(reasoning)}",
        ]
        if needs_plot_summary:
            content.append(
                "Plot overviews:\n"
                + ("\n".join(overview_lines) if overview_lines else "(no overviews)")
            )
        if rec_lines:
            content.append("Recommendations (ratings-aware):\n" + "\n".join(rec_lines))

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n\n".join(content)},
            ],
        )
        return resp.choices[0].message.content or ""
