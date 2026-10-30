"""The closer read for posts Jev is unsure about: a local LLM (Ollama, $0) returns a structured verdict.

One call at a time: CPU inference doesn't parallelize, so the arq worker's 4 job slots would only queue behind
each other inside Ollama and trip timeouts. The first call after a cold start takes minutes (the worker's
job_timeout covers it); a warm call is about 8 s.
"""
import asyncio

import httpx
import ollama
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, field_validator

from triage.config import settings
from triage.errors import PermanentError, RetryableError
from triage.services import jev

MAX_CHARS = 6_000      # a 7B model on CPU: keep the prompt well inside num_ctx
NUM_CTX = 8192         # Ollama's default context is small and silently truncates long prompts
TRIES = 2


class DeepRead(BaseModel):
    fit: bool = Field(description="True if this job is a good fit for the candidate described in the instructions")
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are of the fit verdict, 0 to 1")
    reasons: list[str] = Field(default_factory=list, description="One short reason per line, quoting the post where possible")
    red_flags: list[str] = Field(default_factory=list, description="Anything that makes the role a poor fit or risky")

    @field_validator("confidence", mode="before")
    @classmethod
    def _percent_to_fraction(cls, v):
        # qwen2.5:7b answers "confidence": 100 even when told 0 to 1. A whole number from 2 to 100 is a percentage.
        return v / 100 if isinstance(v, (int, float)) and 2 <= v <= 100 else v


_FIT = jev.QUESTIONS["fit"]
SYSTEM_PROMPT = (
    "You screen job posts for one candidate.\n"
    f"Question: {_FIT['instructions']}\n"
    f"Answer fit=true when: {_FIT['criteria']['true']}\n"
    f"Answer fit=false when: {_FIT['criteria']['false']}\n"
    "Read the whole post, including the location and any note about who may apply. Judge only what the post "
    "says; if it is silent about where the work can be done, say so in red_flags and lower your confidence. "
    "Give at most three short reasons and quote the post where you can. "
    "confidence is a fraction between 0 and 1 (for example 0.85), never a percentage. "
    "Always include red_flags, as an empty list if there are none."
)

_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _one_at_a_time() -> asyncio.Semaphore:
    return _semaphores.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(1))


def _chain():
    llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_URL, temperature=0, num_ctx=NUM_CTX,
                     client_kwargs={"timeout": 600})
    return llm.with_structured_output(DeepRead)


def _human(post: dict) -> str:
    head = "\n".join(f"{k.capitalize()}: {post.get(k) or ''}" for k in ("title", "company", "location"))
    return f"{head}\n\n{(post.get('description_text') or '')[:MAX_CHARS]}"


async def deep_read(post: dict) -> DeepRead:
    messages = [("system", SYSTEM_PROMPT), ("human", _human(post))]
    async with _one_at_a_time():
        chain = _chain()
        for attempt in range(TRIES):
            try:
                out = await chain.ainvoke(messages)
            except (ValidationError, OutputParserException):
                out = None                      # the model wrote something that isn't the schema: try once more
            except ollama.ResponseError as e:
                if e.status_code >= 500:
                    raise RetryableError(f"Ollama {e.status_code}: {e.error}") from e
                raise PermanentError(f"Ollama {e.status_code}: {e.error}") from e
            except (ConnectionError, TimeoutError, httpx.TransportError) as e:    # ollama's client raises httpx errors
                raise RetryableError(f"Ollama unreachable at {settings.OLLAMA_URL}: {e}") from e
            if isinstance(out, DeepRead):
                return out
    raise PermanentError(f"deep read produced invalid output {TRIES} times")
