import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import random

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import config
import dotenv


logger = logging.getLogger(__name__)
CACHE_PATH = Path("cache/llm_cache.jsonl")


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


def build_user_prompt(overview: str) -> str:
    return f"""
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


def make_cache_key(model: str, system_prompt: str, user_prompt: str, response_format: str) -> str:
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_format": response_format,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("key")
            value = record.get("value")
            if key and isinstance(value, dict):
                cache[key] = value
    return cache


def append_cache(path: Path, entries: list[tuple[str, dict]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key, value in entries:
            record = {"key": key, "value": value}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


async def _get_neighborhood_tags_async(
    neighborhood_overviews: list[str],
) -> list[NeighborhoodTags]:
    if not neighborhood_overviews:
        return []
    api_key = os.getenv("NEBIUS_API_KEY")
    assert api_key, "NEBIUS_API_KEY is not set in environment variables"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.NEBIUS_BASE_URL,
    )
    concurrent_requests = getattr(config, "CONCURRENT_REQUESTS", 10)
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def score_one(overview: str) -> NeighborhoodTags:
        user_prompt = build_user_prompt(overview)

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
    if not neighborhood_overviews:
        return []

    cache = load_cache(CACHE_PATH)
    response_format = NeighborhoodTags.__name__
    results: list[NeighborhoodTags | None] = [None] * len(neighborhood_overviews)
    missing: list[tuple[int, str, str]] = []

    for idx, overview in enumerate(neighborhood_overviews):
        user_prompt = build_user_prompt(overview)
        key = make_cache_key(config.NEBIUS_MODEL, SYSTEM_PROMPT, user_prompt, response_format)
        cached = cache.get(key)
        if cached is not None:
            results[idx] = NeighborhoodTags(**cached)
        else:
            missing.append((idx, overview, key))

    if missing:
        dotenv.load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)
        new_overviews = [overview for _, overview, _ in missing]
        new_results = _run_async(_get_neighborhood_tags_async(new_overviews))
        new_cache_entries: list[tuple[str, dict]] = []

        for (idx, _overview, key), tag in zip(missing, new_results):
            results[idx] = tag
            if key not in cache:
                new_cache_entries.append((key, tag.model_dump()))

        append_cache(CACHE_PATH, new_cache_entries)

    return [tag for tag in results if tag is not None]


def add_neighborhood_tags(df):
    if "neighborhood_overview" not in df.columns:
        return df

    overviews = df["neighborhood_overview"].fillna("").astype(str).tolist()
    default_tag = NeighborhoodTags(
        central=0.5,
        many_stores=0.5,
        crowded=0.5,
        noisy=0.5,
        quiet=0.5,
    )

    valid_indices = [i for i, overview in enumerate(overviews) if overview.strip()]
    valid_overviews = [overviews[i] for i in valid_indices]

    tags: list[NeighborhoodTags] = [default_tag] * len(overviews)
    if valid_overviews:
        resolved_tags = get_neighborhood_tags(valid_overviews)
        for idx, tag in zip(valid_indices, resolved_tags):
            tags[idx] = tag

    df = df.copy()
    df["neighborhood_central"] = [tag.central for tag in tags]
    df["neighborhood_many_stores"] = [tag.many_stores for tag in tags]
    df["neighborhood_crowded"] = [tag.crowded for tag in tags]
    df["neighborhood_noisy"] = [tag.noisy for tag in tags]
    df["neighborhood_quiet"] = [tag.quiet for tag in tags]
    return df
