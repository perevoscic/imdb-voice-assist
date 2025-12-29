import json
import os
from typing import Dict, List, Optional

from openai import OpenAI


def _format_conversation_for_response(messages: List[Dict], max_turns: int = 3) -> str:
    """Format recent conversation history for response generation."""
    if not messages:
        return ""
    
    # Get last N turns
    recent = messages[-(max_turns * 2):]
    
    formatted = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            formatted.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted)


def build_response(
    client: OpenAI,
    query: str,
    results: List[dict],
    recommendations: List[dict],
    conversation_history: Optional[List[Dict]] = None,
    min_year: int = 1920,
    max_year: int = 2020,
) -> str:
    lowered = query.lower()
    if ("release" in lowered or "released" in lowered) and results:
        movie = results[0]
        year = movie.get("Released_Year")
        if year:
            if isinstance(year, float) and year.is_integer():
                year = int(year)
            return str(year)
    
    history_str = _format_conversation_for_response(conversation_history or [])
    
    system = (
        "You answer questions about IMDB movies using ONLY the provided data. "
        "Do not add facts that are not explicitly present in the data. "
        "If a specific detail (like an exact release date) is missing, say that it is not available "
        "and provide the closest available information (e.g., Released_Year). "
        "Include a concise answer, brief reasoning, and 1-2 suggestions for follow-up questions. "
        "If asked to summarize plots, summarize using Overview fields. "
        "If results are empty, say so and suggest a refined query. "
        "Use conversation history to understand context and maintain a natural conversational flow. "
        "When the user refers to previous topics (e.g., 'that movie', 'tell me more'), "
        "acknowledge the context naturally in your response. "
        f"IMPORTANT: The database contains movies from {min_year} to {max_year}. "
        "When no specific time period is mentioned, results are from the FULL available range. "
        "Do NOT ask clarifying questions about time periods or year ranges - "
        "just provide the answer based on all available data."
    )
    
    payload = {
        "conversation_history": history_str if history_str else None,
        "current_query": query,
        "results": results,
        "recommendations": recommendations,
    }
    
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content
