import asyncio

import ollama
import pytest
from pydantic import ValidationError

from triage import errors
from triage.services import deep_read
from triage.services.deep_read import DeepRead

POST = {"title": "ML Engineer", "company": "Acme", "location": "Remote", "description_text": "Build LLM things."}
GOOD = DeepRead(fit=True, confidence=0.8, reasons=["remote, junior"], red_flags=[])


class FakeChain:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.live, self.peak = list(outcomes), [], 0, 0

    async def ainvoke(self, messages):
        self.calls.append(messages)
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0.01)
        self.live -= 1
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def chain(monkeypatch):
    def install(*outcomes):
        c = FakeChain(outcomes)
        monkeypatch.setattr(deep_read, "_chain", lambda: c)
        return c
    return install


def test_schema_bounds_confidence():
    for bad in (1.5, 150, -0.1):
        with pytest.raises(ValidationError):
            DeepRead(fit=True, confidence=bad, reasons=[], red_flags=[])
    assert DeepRead(fit=False, confidence=0.0).reasons == []


def test_percentage_confidence_from_the_model_is_scaled():
    # Observed live: qwen2.5:7b returned "confidence": 100 and left red_flags out entirely.
    d = DeepRead.model_validate({"fit": True, "confidence": 100, "reasons": ["remote"]})
    assert (d.confidence, d.red_flags) == (1.0, [])
    assert DeepRead(fit=True, confidence=85).confidence == 0.85
    assert DeepRead(fit=True, confidence=1).confidence == 1.0


async def test_returns_the_structured_verdict(chain):
    c = chain(GOOD)
    assert await deep_read.deep_read(POST) == GOOD
    system, human = c.calls[0]
    assert "Pakistan" in system[1] and "ML Engineer" in human[1] and "Build LLM things." in human[1]


async def test_prompt_reuses_the_jev_criteria_so_the_two_stages_agree(chain):
    from triage.services import jev
    chain(GOOD)
    assert jev.QUESTIONS["fit"]["criteria"]["true"] in deep_read.SYSTEM_PROMPT
    assert jev.QUESTIONS["fit"]["criteria"]["false"] in deep_read.SYSTEM_PROMPT


async def test_description_is_capped(chain):
    c = chain(GOOD)
    await deep_read.deep_read({**POST, "description_text": "x" * 100_000})
    assert len(c.calls[0][1][1]) < deep_read.MAX_CHARS + 500


async def test_one_ollama_call_at_a_time(chain):
    c = chain(GOOD)
    await asyncio.gather(*(deep_read.deep_read(POST) for _ in range(5)))
    assert len(c.calls) == 5 and c.peak == 1          # CPU inference doesn't parallelize


async def test_malformed_model_output_is_retried_once_then_permanent(chain):
    bad = ValidationError.from_exception_data("DeepRead", [])
    c = chain(bad, GOOD)
    assert await deep_read.deep_read(POST) == GOOD and len(c.calls) == 2
    c = chain(bad)
    with pytest.raises(errors.PermanentError, match="invalid"):
        await deep_read.deep_read(POST)
    assert len(c.calls) == 2


async def test_none_from_structured_output_counts_as_malformed(chain):
    c = chain(None, GOOD)                              # with_structured_output can return None on a parse failure
    assert await deep_read.deep_read(POST) == GOOD and len(c.calls) == 2


async def test_ollama_down_is_retryable(chain):
    chain(ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(errors.RetryableError):
        await deep_read.deep_read(POST)


async def test_ollama_server_error_retryable_missing_model_permanent(chain):
    chain(ollama.ResponseError("overloaded", 503))
    with pytest.raises(errors.RetryableError):
        await deep_read.deep_read(POST)
    chain(ollama.ResponseError("model 'qwen2.5:7b' not found", 404))
    with pytest.raises(errors.PermanentError, match="not found"):
        await deep_read.deep_read(POST)


async def test_httpx_connection_and_timeout_errors_are_retryable_too(chain):
    # Found live: with Ollama not running, ollama's client raised httpx.ConnectError, not ConnectionError.
    import httpx
    for exc in (httpx.ConnectError("All connection attempts failed"), httpx.ReadTimeout("slow cold start")):
        chain(exc)
        with pytest.raises(errors.RetryableError):
            await deep_read.deep_read(POST)
