import json

import httpx
import pytest
import respx

from triage import errors
from triage.config import settings
from triage.services import jev

URL = "https://api.typesafe.ai/v1/systemone"
OK = {"model": "jev-1.13.0", "answers": {"fit": {"type": "noul", "noul": 0.83}}, "usage": {"input_tokens": 400}}


@pytest.fixture(autouse=True)
def key_and_fast_sleep(monkeypatch):
    monkeypatch.setattr(settings, "JEV_API_KEY", "test-key")
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(jev, "_sleep", fake_sleep)
    return slept


def test_questions_have_the_shape_jev_needs():
    q = jev.QUESTIONS["fit"]
    assert q["instructions"] and set(q["criteria"]) == {"true", "false"}


def test_build_request_matches_the_wire_format():
    body = jev.build_request("some post", jev.QUESTIONS)
    assert body["state"] == "some post" and body["model"] == settings.JEV_MODEL
    assert body["questions"]["fit"]["type"] == "noul" and "instructions" in body["questions"]["fit"]


@respx.mock
async def test_ask_returns_probabilities_and_sends_auth():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=OK))
    assert await jev.ask("post text") == {"fit": 0.83}
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer test-key"
    assert json.loads(req.content)["state"] == "post text"


@respx.mock
async def test_ask_with_usage_reports_resolved_model_and_cost():
    respx.post(URL).mock(return_value=httpx.Response(200, json=OK))
    probs, usage = await jev.ask_with_usage("t")
    assert probs == {"fit": 0.83} and usage == {"input_tokens": 400, "model": "jev-1.13.0"}
    assert jev.cost_usd(usage) == pytest.approx(400 * 0.042 / 1_000_000)


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 503, 408])
async def test_transient_errors_are_retried_with_backoff(status, key_and_fast_sleep):
    route = respx.post(URL).mock(side_effect=[httpx.Response(status), httpx.Response(status), httpx.Response(200, json=OK)])
    assert await jev.ask("t") == {"fit": 0.83}
    assert route.call_count == 3 and key_and_fast_sleep == [1, 2]


@respx.mock
async def test_gives_up_with_a_retryable_error_after_four_tries(key_and_fast_sleep):
    route = respx.post(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(errors.RetryableError):
        await jev.ask("t")
    assert route.call_count == 4


@respx.mock
async def test_network_errors_are_retried():
    respx.post(URL).mock(side_effect=[httpx.ConnectError("down"), httpx.Response(200, json=OK)])
    assert await jev.ask("t") == {"fit": 0.83}


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 422])
async def test_client_errors_are_permanent_and_not_retried(status):
    route = respx.post(URL).mock(return_value=httpx.Response(status, text="nope"))
    with pytest.raises(errors.PermanentError, match=str(status)):
        await jev.ask("t")
    assert route.call_count == 1


@respx.mock
async def test_malformed_response_is_permanent():
    respx.post(URL).mock(return_value=httpx.Response(200, json={"oops": 1}))
    with pytest.raises(errors.PermanentError):
        await jev.ask("t")
    respx.post(URL).mock(return_value=httpx.Response(200, json={"answers": {"fit": {"noul": "high"}}}))
    with pytest.raises(errors.PermanentError):
        await jev.ask("t")


async def test_missing_api_key_is_permanent(monkeypatch):
    monkeypatch.setattr(settings, "JEV_API_KEY", "")
    with pytest.raises(errors.PermanentError, match="JEV_API_KEY"):
        await jev.ask("t")


def test_post_text_has_title_company_location_and_description_and_is_capped():
    post = {"title": "ML Engineer", "company": "Acme", "location": "Remote", "remote_eligibility": "eligible",
            "description_text": "x" * 50_000}
    text = jev.post_text(post)
    assert text.startswith("ML Engineer\nAcme\nRemote") and len(text) < 13_000
