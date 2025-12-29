import json
import os
from typing import List

from openai import OpenAI


def build_response(
    client: OpenAI,
    query: str,
    results: List[dict],
    recommendations: List[dict],
) -> str:
    system = (
        "You answer questions about IMDB movies using ONLY the provided data. "
        "Do not add facts that are not explicitly present in the data. "
        "If a specific detail (like an exact release date) is missing, say that it is not available "
        "and provide the closest available information (e.g., Released_Year). "
        "Include a concise answer, brief reasoning, and 1-2 suggestions. "
        "If asked to summarize plots, summarize using Overview fields. "
        "If results are empty, say so and suggest a refined query."
    )
    payload = {
        "query": query,
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
