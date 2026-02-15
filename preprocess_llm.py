import os

from openai import OpenAI
from pydantic import BaseModel, Field

from config import config


class NeighborhoodTags(BaseModel):
    central: float = Field(ge=0, le=1)
    many_stores: float = Field(ge=0, le=1)
    crowded: float = Field(ge=0, le=1)
    noisy: float = Field(ge=0, le=1)
    quiet: float = Field(ge=0, le=1)


SYSTEM_PROMPT = """You are an expert analyst of guest perception for rental listings.
Analyze the neighborhood overview text.
Return scores between 0 and 1 for each category: central, many stores, crowded, noisy, quiet.
If you don't know, answer with 0.5 ."""


def get_neighborhood_tags(neighborhood_overviews: list[str]) -> list[NeighborhoodTags]:
    client = OpenAI(
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=config.NEBIUS_BASE_URL,
    )
    results: list[NeighborhoodTags] = []
    for overview in neighborhood_overviews:
        user_prompt = f"""
Neighborhood Overview Begin
{overview}
Neighborhood Overview End

Return JSON only as:
{{
  \"central\": <float>,
  \"many_stores\": <float>,
  \"crowded\": <float>,
  \"noisy\": <float>,
  \"quiet\": <float>
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.parse(
            model=config.NEBIUS_MODEL,
            messages=messages,
            response_format=NeighborhoodTags,
        )
        results.append(response.choices[0].message.parsed)
    return results
