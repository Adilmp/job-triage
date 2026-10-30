"""The only module that knows what a Jev API call looks like (async port of jev-calibration/jev.py).

Jev takes a `state` (the post) and a dict of yes/no questions and returns a probability for each. The raw
probability is used directly: there is no calibration step.

QUESTIONS is the screening criteria. WRITE AND FREEZE THE WORDING BEFORE LOOKING AT ANY EVAL NUMBERS: it is
part of criteria_version(), so editing it later re-triages old posts, and tuning it against the held-out
split would defeat the split.
"""
import asyncio

import httpx

from triage.config import settings
from triage.errors import PermanentError, RetryableError

ENDPOINT = "/v1/systemone"
PRICE_PER_MTOK = 0.042          # USD; from the 8,000-judgment calibration run
MAX_CHARS = 12_000              # descriptions are ~2-3k tokens; cap the outliers
RETRIES = 4                     # 1s, 2s, 4s between tries

# DRAFT wording: the user owns this. "fit" must match the one-sentence definition they write before labeling.
QUESTIONS = {
    "fit": {
        "instructions": (
            "Would this job be a good fit for an early-career AI/LLM or backend software engineer who lives "
            "in Pakistan, will not relocate, and wants to work either remotely or from an office in Pakistan?"
        ),
        "criteria": {
            "true": (
                "A software, backend, ML, LLM/GenAI (including fine-tuning) or data engineering role at junior to "
                "mid level that can be done from Pakistan: fully remote (the post says remote worldwide, or remote in a "
                "region that includes Pakistan), or based in an office in Pakistan (Karachi, Lahore, Islamabad), "
                "on-site or hybrid. A senior-titled "
                "role also counts when the post asks for only about four years of experience or less."
            ),
            "false": (
                "A staff, principal or leadership role, or any role asking for about six or more years of "
                "experience; a non-engineering role (sales, marketing, operations, "
                "recruiting, finance); a role that needs on-site presence or relocation outside Pakistan; "
                "remote only inside a region that excludes Pakistan (for example US-only or EU-only); or a "
                "role that requires work authorization somewhere else."
            ),
        },
    },
}


async def _sleep(seconds: float) -> None:        # a seam so tests don't wait
    await asyncio.sleep(seconds)


def build_request(text: str, questions: dict) -> dict:
    return {
        "model": settings.JEV_MODEL,
        "state": text,
        "questions": {name: {"type": "noul", **spec} for name, spec in questions.items()},
    }


def parse_response(payload: dict, questions: dict) -> dict[str, float]:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise PermanentError(f"Jev response has no 'answers' object: {str(payload)[:300]}")
    out = {}
    for name in questions:
        item = answers.get(name)
        if isinstance(item, dict) and isinstance(item.get("noul"), (int, float)):
            out[name] = float(item["noul"])
    if not out:
        raise PermanentError(f"Jev response has no probabilities for {list(questions)}: {str(payload)[:300]}")
    return out


def post_text(post: dict) -> str:
    """What Jev reads: title, company, location, then the description."""
    head = "\n".join(str(post.get(k) or "") for k in ("title", "company", "location"))
    return f"{head}\n\n{(post.get('description_text') or '')[:MAX_CHARS]}"


async def ask_with_usage(text: str, questions: dict = QUESTIONS, client: httpx.AsyncClient | None = None):
    """({question: probability}, usage). usage carries the real token counts and the model version the alias
    resolved to ('jev-latest' -> 'jev-1.13.0'), so cost and reproducibility come from what was billed."""
    if not settings.JEV_API_KEY:
        raise PermanentError("JEV_API_KEY is not set (see .env.example)")
    body = build_request(text, questions)
    headers = {"Authorization": f"Bearer {settings.JEV_API_KEY}"}
    url = settings.JEV_BASE_URL.rstrip("/") + ENDPOINT

    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                r = await client.post(url, json=body, headers=headers)
            except httpx.InvalidURL as e:
                raise PermanentError(f"invalid Jev URL {url}") from e
            except httpx.HTTPError as e:
                last = RetryableError(f"{type(e).__name__} calling Jev")
            else:
                if r.status_code == 200:
                    try:
                        payload = r.json()
                    except ValueError as e:
                        raise PermanentError("Jev returned non-JSON") from e
                    usage = dict(payload.get("usage") or {})
                    usage["model"] = payload.get("model")
                    return parse_response(payload, questions), usage
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise PermanentError(f"Jev HTTP {r.status_code}: {r.text[:300]}")      # retrying won't help
                last = RetryableError(f"Jev HTTP {r.status_code}")
            if attempt < RETRIES - 1:
                await _sleep(2 ** attempt)
        raise last
    finally:
        if own:
            await client.aclose()


async def ask(text: str, questions: dict = QUESTIONS, client: httpx.AsyncClient | None = None) -> dict[str, float]:
    return (await ask_with_usage(text, questions, client))[0]


def cost_usd(usage: dict) -> float:
    return (usage.get("input_tokens") or 0) * PRICE_PER_MTOK / 1_000_000
