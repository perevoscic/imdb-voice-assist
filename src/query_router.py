import json
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from src.config import OPENAI_API_KEY, OPENAI_MODEL


class RouterOutput(BaseModel):
    intent: str = Field(description="structured | semantic | hybrid")
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort: Optional[Dict[str, str]] = None
    groupby: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    semantic_query: Optional[str] = None
    clarifying_question: Optional[str] = None


SYSTEM_PROMPT = """
You are a query router for an IMDB dataset. Return ONLY valid JSON.
Fields:
- intent: structured | semantic | hybrid
- filters: year_min/year_max, genre, imdb_min, metascore_min, votes_min, gross_min, director, title
- sort: {column, order}
- groupby: {column, agg, metric}
- limit: integer
- semantic_query: string for similarity search over Overview
- clarifying_question: string if query is ambiguous

Rules:
- If the user asks about plot themes or narrative elements, use semantic or hybrid.
- Use hybrid when a structured filter is combined with plot similarity.
- Keep filters numeric where possible.
""".strip()


class QueryRouter:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def route(self, user_text: str) -> RouterOutput:
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
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
