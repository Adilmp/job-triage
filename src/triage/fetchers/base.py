import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from triage.errors import PermanentError, RetryableError  # noqa: F401  (re-exported for the fetchers)

UA = "Mozilla/5.0 (compatible; job-triage/0.1)"
TIMEOUT = 15


@dataclass
class RawPost:
    provider: str
    external_id: str
    url: str
    title: str
    location: str | None
    description_text: str      # plain text; may be empty, ingest drops those loudly (P2)
    raw: dict = field(repr=False)   # the provider's payload for this job, kept as-is (P4)


Fetcher = Callable[[httpx.AsyncClient, str], Awaitable[list[RawPost]]]


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"User-Agent": UA}, timeout=TIMEOUT, follow_redirects=True)


async def get_text(client: httpx.AsyncClient, url: str, params: dict | None = None) -> str:
    try:
        r = await client.get(url, params=params)
    except httpx.InvalidURL as e:
        raise PermanentError(f"invalid URL {url}") from e
    except httpx.HTTPError as e:
        raise RetryableError(f"{type(e).__name__} fetching {url}") from e
    if r.status_code == 429:
        retry_after = r.headers.get("retry-after", "")
        raise RetryableError(f"429 from {url}", int(retry_after) if retry_after.isdigit() else None)
    if r.status_code >= 500:
        raise RetryableError(f"{r.status_code} from {url}")
    if r.status_code != 200:
        raise PermanentError(f"{r.status_code} from {url}")
    return r.text


async def get_json(client: httpx.AsyncClient, url: str, params: dict | None = None):
    text = await get_text(client, url, params)
    try:
        return json.loads(text)
    except ValueError as e:
        raise PermanentError(f"{url} did not return JSON") from e
