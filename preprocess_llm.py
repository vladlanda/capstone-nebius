import asyncio
import concurrent.futures
import hashlib
import logging
import os
import random

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import config


logger = logging.getLogger(__name__)


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


async def _get_neighborhood_tags_async(
    neighborhood_overviews: list[str],
) -> list[NeighborhoodTags]:
    if not neighborhood_overviews:
        return []

    client = AsyncOpenAI(
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=config.NEBIUS_BASE_URL,
    )
    concurrent_requests = getattr(config, "CONCURRENT_REQUESTS", 10)
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def score_one(overview: str) -> NeighborhoodTags:
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

        max_attempts = 5
        query_hash = hashlib.sha256(overview.encode("utf-8")).hexdigest()
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.parse(
                    model=config.NEBIUS_MODEL,
                    messages=messages,
                    response_format=NeighborhoodTags,
                )
                return response.choices[0].message.parsed
            except Exception:
                logger.warning(
                    "Neighborhood tag request failed. query_hash=%s attempt=%d/%d",
                    query_hash,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep((2 ** attempt) + random.random())

        return NeighborhoodTags(
            central=0.5,
            many_stores=0.5,
            crowded=0.5,
            noisy=0.5,
            quiet=0.5,
        )

    async def sem_task(overview: str) -> NeighborhoodTags:
        async with semaphore:
            return await score_one(overview)

    tasks = [asyncio.create_task(sem_task(overview)) for overview in neighborhood_overviews]
    return await asyncio.gather(*tasks)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)


def get_neighborhood_tags(neighborhood_overviews: list[str]) -> list[NeighborhoodTags]:
    return _run_async(_get_neighborhood_tags_async(neighborhood_overviews))
