from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI


REFUSAL_MESSAGE = (
    "I can't help with that. Ask me about movies, ratings, directors, or plots "
    "from the IMDb dataset."
)
OUT_OF_SCOPE_MESSAGE = (
    "I'm an IMDb assistant. Ask about movies, directors, ratings, years, genres, plots."
)
PROFANITY_NUDGE = "I can help with IMDb questions. Let's keep it respectful."

MOVIE_HINTS = (
    "movie",
    "movies",
    "film",
    "films",
    "imdb",
    "director",
    "directors",
    "actor",
    "actors",
    "actress",
    "cast",
    "genre",
    "genres",
    "plot",
    "plots",
    "summary",
    "summaries",
    "overview",
    "rating",
    "ratings",
    "score",
    "metascore",
    "release",
    "released",
    "year",
    "runtime",
    "box office",
    "gross",
    "superhero",
    "super hero",
    "superheroes",
    "super heroes",
)

OUT_OF_SCOPE_HINTS = (
    "self-harm",
    "suicide",
    "kill myself",
    "weapon",
    "bomb",
    "drugs",
    "porn",
    "child porn",
    "doxx",
)

PROFANITY_WORDS = (
    "damn",
    "shit",
    "bullshit",
    "fuck",
    "fucking",
    "bitch",
    "asshole",
    "bastard",
    "crap",
)
PROFANITY_RE = re.compile(rf"\\b({'|'.join(PROFANITY_WORDS)})\\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


@dataclass
class ModerationResult:
    flagged: bool
    categories: Dict[str, bool]


@dataclass
class ScopeResult:
    in_scope: bool
    used_llm: bool


def _history_in_scope(conversation_history: List[Dict]) -> bool:
    for message in reversed(conversation_history):
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        heuristic = _heuristic_scope(content)
        return heuristic is True
    return False


def moderate_text(client: OpenAI, text: str) -> ModerationResult:
    if not text or not text.strip():
        return ModerationResult(flagged=False, categories={})
    try:
        response = client.moderations.create(
            model=os.getenv("MODERATION_MODEL", "omni-moderation-latest"),
            input=text,
        )
        result = response.results[0]
        categories = {}
        if hasattr(result, "categories"):
            cat_obj = result.categories
            if hasattr(cat_obj, "model_dump"):
                categories = cat_obj.model_dump()
            else:
                categories = dict(cat_obj)
        
        is_flagged = bool(result.flagged)
        
        # False positive check: Allow "violence" categories if the query is clearly about movies
        # (e.g. "Summarize movies with Al Pacino" often triggers violence flags incorrectly)
        if is_flagged:
            # Check if ONLY permissible categories are triggered
            permissible = {"violence", "violence/graphic"}
            triggered = {k for k, v in categories.items() if v}
            
            if triggered and triggered.issubset(permissible):
                # Check if it looks like a movie query (use full scope check including LLM)
                # We pass None for history as we only care about the current query's safety context
                if is_in_scope(client, text, conversation_history=None).in_scope:
                    is_flagged = False

        return ModerationResult(flagged=is_flagged, categories=categories)
    except Exception:
        return ModerationResult(flagged=False, categories={})  # Fail open on error



def _heuristic_scope(text: str) -> Optional[bool]:
    lowered = text.lower()
    if any(hint in lowered for hint in MOVIE_HINTS):
        return True
    if any(hint in lowered for hint in OUT_OF_SCOPE_HINTS):
        return False
    if YEAR_RE.search(lowered):
        return True
    return None


def is_in_scope(
    client: OpenAI, text: str, conversation_history: Optional[List[Dict]] = None
) -> ScopeResult:
    heuristic = _heuristic_scope(text)
    if heuristic is not None:
        return ScopeResult(in_scope=heuristic, used_llm=False)

    if conversation_history and _history_in_scope(conversation_history):
        return ScopeResult(in_scope=True, used_llm=False)

    system = (
        "You are a classifier. Determine if the user's message is about movies or "
        "the IMDb dataset (movies, directors, actors, ratings, plots, genres, years). "
        "Answer only 'yes' or 'no'."
    )
    history_text = ""
    if conversation_history:
        recent = conversation_history[-5:]
        history_lines = [
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent
        ]
        history_text = "Conversation history:\n" + "\n".join(history_lines) + "\n\n"
    try:
        response = client.chat.completions.create(
            model=os.getenv("SCOPE_MODEL", os.getenv("CHAT_MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{history_text}Current message: {text}"},
            ],
            max_tokens=5,
            temperature=0,
        )
        content = response.choices[0].message.content.strip().lower()
        return ScopeResult(in_scope=content.startswith("y"), used_llm=True)
    except Exception:
        return ScopeResult(in_scope=True, used_llm=False)


def contains_profanity(text: str) -> bool:
    if not text:
        return False
    return bool(PROFANITY_RE.search(text.lower()))


def safety_refusal_message() -> str:
    return REFUSAL_MESSAGE


def out_of_scope_message() -> str:
    return OUT_OF_SCOPE_MESSAGE


def profanity_nudge_message() -> str:
    return PROFANITY_NUDGE
