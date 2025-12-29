import json
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from src.config import OPENAI_API_KEY, OPENAI_MODEL


class RouterOutput(BaseModel):
    intent: str = Field(description="structured | semantic | hybrid | analytic")
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort: Optional[Dict[str, str]] = None
    groupby: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    semantic_query: Optional[str] = None
    clarifying_question: Optional[str] = None
    analytic_type: Optional[str] = None
    needs_plot_summary: Optional[bool] = None


SYSTEM_PROMPT = """
You are a query router for an IMDB dataset. Return ONLY valid JSON.
Fields:
- intent: structured | semantic | hybrid | analytic
- filters: year_min/year_max, genre, imdb_min, metascore_min, votes_min, gross_min,
  director, title, actor, actor_role
- sort: {column, order}
- groupby: {column, agg, metric}
- limit: integer
- semantic_query: string for similarity search over Overview
- clarifying_question: string if query is ambiguous
- analytic_type: string for specialized aggregations
- needs_plot_summary: boolean

Rules:
- If the user asks about plot themes or narrative elements, use semantic or hybrid.
- Use hybrid when a structured filter is combined with plot similarity.
- Use analytic intent for specialized aggregations not covered by groupby.
- Keep filters numeric where possible.
- "before YEAR" -> year_max = YEAR - 1. "between YEAR1-YEAR2" -> year_min/year_max.
- "top-rated" -> sort by IMDB_Rating desc.
- "top by meta score" -> sort by Meta_score desc.
- "lower gross" -> sort by Gross_num asc.
- If a year is given like "in 2019", set year_min/year_max to that year.
- Sort columns should match dataframe fields (Released_Year, IMDB_Rating, Meta_score, No_of_votes, Gross_num).
- If user asks about actor roles (lead vs any), set clarifying_question.
- If user clarifies lead, set actor_role = "lead" (use Star1). Otherwise actor_role = "any".
- If the query is "top directors and their highest grossing movies with gross > X at least N times",
  set intent = "analytic", analytic_type = "director_gross_min_count",
  filters.gross_min = X, filters.min_count = N.
- If user wants plot summaries, set needs_plot_summary = true.
""".strip()


class QueryRouter:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)

    def route(self, user_text: str) -> RouterOutput:
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {}
        try:
            return RouterOutput(**payload)
        except ValidationError:
            return RouterOutput(intent="structured")
