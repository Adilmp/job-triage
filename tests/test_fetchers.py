import copy
import json
from pathlib import Path

import httpx
import pytest
import respx

from triage.derive import derive
from triage.fetchers import FETCHERS
from triage.fetchers.base import PermanentError, RawPost, RetryableError, get_json, make_client

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


@pytest.fixture
async def client():
    async with make_client() as c:
        yield c


# ---------- base: error mapping ----------

@respx.mock
@pytest.mark.parametrize("status", [404, 403, 401])
async def test_client_errors_are_permanent(client, status):
    respx.get("https://x.test/a").mock(return_value=httpx.Response(status))
    with pytest.raises(PermanentError):
        await get_json(client, "https://x.test/a")


@respx.mock
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_server_errors_are_retryable(client, status):
    respx.get("https://x.test/a").mock(return_value=httpx.Response(status))
    with pytest.raises(RetryableError):
        await get_json(client, "https://x.test/a")


@respx.mock
async def test_429_is_retryable_and_carries_retry_after(client):
    respx.get("https://x.test/a").mock(return_value=httpx.Response(429, headers={"retry-after": "86000"}))
    with pytest.raises(RetryableError) as e:
        await get_json(client, "https://x.test/a")
    assert e.value.retry_after == 86000      # the worker must not hammer a site that asked for a day (Workable did)


@respx.mock
async def test_timeouts_and_connection_errors_are_retryable(client):
    respx.get("https://x.test/a").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(RetryableError):
        await get_json(client, "https://x.test/a")
    respx.get("https://x.test/b").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(RetryableError):
        await get_json(client, "https://x.test/b")


@respx.mock
async def test_non_json_200_is_permanent(client):
    respx.get("https://x.test/a").mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(PermanentError):
        await get_json(client, "https://x.test/a")


async def test_registry_has_all_six_providers():
    assert set(FETCHERS) == {"greenhouse", "lever", "ashby", "recruitee", "workable", "teamtailor"}


# ---------- Greenhouse (P1) ----------

GH_URL = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"


@respx.mock
async def test_greenhouse_asks_for_content_and_cleans_escaped_html(client):
    route = respx.get(GH_URL).mock(return_value=httpx.Response(200, json=load("greenhouse")))
    posts = await FETCHERS["greenhouse"](client, "acme")
    assert route.calls[0].request.url.params["content"] == "true"
    assert len(posts) == 2 and all(isinstance(p, RawPost) for p in posts)
    p = posts[0]
    assert (p.provider, p.external_id) == ("greenhouse", "5456619008")
    assert p.title == "3P Revenue Accounting & Reporting"
    assert p.url == "https://job-boards.greenhouse.io/anthropic/jobs/5456619008"
    assert p.location == "San Francisco, CA | Seattle, WA"
    assert "About Anthropic" in p.description_text and "Anthropic’s mission" in p.description_text
    assert not any(t in p.description_text for t in ("&lt;", "&gt;", "&quot;", "<div", "<p>"))
    assert p.raw["metadata"][0]["value"] == "On-Site"                 # raw payload kept (P4)
    assert derive("greenhouse", p.raw, p.location).remote_eligibility == "onsite"
    assert derive("greenhouse", posts[1].raw, posts[1].location).remote_eligibility == "region_locked"


@respx.mock
async def test_greenhouse_empty_content_is_returned_empty_for_ingest_to_drop(client):
    data = load("greenhouse")
    data["jobs"][0]["content"] = ""
    respx.get(GH_URL).mock(return_value=httpx.Response(200, json=data))
    posts = await FETCHERS["greenhouse"](client, "acme")
    assert len(posts) == 2 and posts[0].description_text == ""        # ingest owns the loud drop (P2)


@respx.mock
async def test_greenhouse_missing_location_and_content_keys(client):
    respx.get(GH_URL).mock(return_value=httpx.Response(200, json={"jobs": [{"id": 1, "title": "T", "absolute_url": "https://u"}]}))
    (p,) = await FETCHERS["greenhouse"](client, "acme")
    assert (p.location, p.description_text) == (None, "")


# ---------- Lever ----------

LEVER_URL = "https://api.lever.co/v0/postings/acme"


@respx.mock
async def test_lever_parses_posts(client):
    route = respx.get(LEVER_URL).mock(return_value=httpx.Response(200, json=load("lever")))
    posts = await FETCHERS["lever"](client, "acme")
    assert route.calls[0].request.url.params["mode"] == "json"
    p = next(x for x in posts if x.external_id.startswith("6ed76ce8"))
    assert p.title == "Administrative Business Partner"
    assert p.url == "https://jobs.lever.co/palantir/6ed76ce8-4156-4b60-b120-403538bd66cd"
    assert p.location == "Singapore, Singapore"
    assert "Palantir builds the world" in p.description_text and "\xa0" not in p.description_text
    assert derive("lever", p.raw, p.location).remote_eligibility == "hybrid"
    # description body + the structured requirement lists + the closing paragraph
    f = next(x for x in posts if x.external_id.startswith("0ccbe620"))
    assert "What We’re Looking For" in f.description_text and "What We Require" in f.description_text
    assert "never be asked to submit a payment" in f.description_text
    assert "<li>" not in f.description_text and "<div" not in f.description_text
    assert derive("lever", f.raw, f.location).remote_eligibility == "region_locked"    # remote, location "North America"


@respx.mock
async def test_lever_joins_multiple_locations(client):
    data = load("lever")
    data[0]["categories"]["allLocations"] = ["Singapore", "London"]
    respx.get(LEVER_URL).mock(return_value=httpx.Response(200, json=data))
    assert (await FETCHERS["lever"](client, "acme"))[0].location == "Singapore | London"


@respx.mock
async def test_lever_unknown_board_is_a_200_error_object_and_permanent(client):
    respx.get(LEVER_URL).mock(return_value=httpx.Response(200, json={"ok": False, "error": "Document not found"}))
    with pytest.raises(PermanentError, match="Document not found"):
        await FETCHERS["lever"](client, "acme")


@respx.mock
async def test_lever_empty_board_is_fine(client):
    respx.get(LEVER_URL).mock(return_value=httpx.Response(200, json=[]))
    assert await FETCHERS["lever"](client, "acme") == []


# ---------- Ashby (P3) ----------

ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/acme"


@respx.mock
async def test_ashby_parses_posts_and_keeps_compensation_payload(client):
    route = respx.get(ASHBY_URL).mock(return_value=httpx.Response(200, json=load("ashby")))
    posts = await FETCHERS["ashby"](client, "acme")
    assert route.calls[0].request.url.params["includeCompensation"] == "true"
    assert len(posts) == 3
    vapi = posts[0]
    assert vapi.title == "Enterprise Account Executive"
    assert vapi.url.startswith("https://jobs.ashbyhq.com/vapi/")
    assert vapi.location == "San Francisco | New York"                  # secondary locations are appended
    assert vapi.description_text and "<" not in vapi.description_text
    d = derive("ashby", vapi.raw, vapi.location)
    assert (d.comp_min, d.comp_max, d.comp_currency) == (150000, 175000, "USD")
    assert d.remote_eligibility == "unclear"                             # isRemote=true but workplaceType=Hybrid


@respx.mock
async def test_ashby_falls_back_to_stripped_html_when_plain_text_missing(client):
    data = load("ashby")
    job = data["jobs"][0]
    html_text = job["descriptionHtml"]
    del job["descriptionPlain"]
    respx.get(ASHBY_URL).mock(return_value=httpx.Response(200, json=data))
    p = (await FETCHERS["ashby"](client, "acme"))[0]
    assert p.description_text and "<" not in p.description_text and html_text.startswith("<")

    job["descriptionPlain"] = ""              # empty string counts as missing too
    respx.get(ASHBY_URL).mock(return_value=httpx.Response(200, json=data))
    assert (await FETCHERS["ashby"](client, "acme"))[0].description_text


@respx.mock
async def test_ashby_accepts_description_key_and_locationName(client):
    job = {"id": "abc", "title": "T", "jobUrl": "https://jobs.ashbyhq.com/acme/abc",
           "description": "<p>Hello <b>there</b></p>", "locationName": "Dubai"}
    respx.get(ASHBY_URL).mock(return_value=httpx.Response(200, json={"jobs": [job]}))
    (p,) = await FETCHERS["ashby"](client, "acme")
    assert (p.description_text, p.location) == ("Hello there", "Dubai")


@respx.mock
async def test_ashby_skips_unlisted_jobs(client):
    data = copy.deepcopy(load("ashby"))
    data["jobs"][0]["isListed"] = False
    respx.get(ASHBY_URL).mock(return_value=httpx.Response(200, json=data))
    assert len(await FETCHERS["ashby"](client, "acme")) == 2


@respx.mock
async def test_ashby_unknown_board_is_permanent(client):
    respx.get(ASHBY_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(PermanentError):
        await FETCHERS["ashby"](client, "acme")


# ---------- Recruitee ----------

REC_URL = "https://acme.recruitee.com/api/offers/"


@respx.mock
async def test_recruitee_parses_offers(client):
    respx.get(REC_URL).mock(return_value=httpx.Response(200, json=load("recruitee")))
    posts = await FETCHERS["recruitee"](client, "acme")
    assert len(posts) == 2
    p = posts[0]
    assert (p.provider, p.external_id) == ("recruitee", "117551")
    assert p.title == "Digital Marketing Specialist - Senior / Junior"
    assert p.url == "https://cognitev.recruitee.com/o/digital-marketing-specialist-senior-junior"
    assert p.location == "6th of October, Cairo, Al Qāhirah, Egypt"
    # description and requirements are separate HTML fields; both end up in the text
    assert "Who are you?" in p.description_text and "Skills" in p.description_text
    assert "<" not in p.description_text
    assert derive("recruitee", p.raw, p.location).remote_eligibility == "onsite"


@respx.mock
async def test_recruitee_skips_unpublished_and_tolerates_missing_requirements(client):
    data = load("recruitee")
    data["offers"][0]["status"] = "draft"
    data["offers"][1]["requirements"] = None
    respx.get(REC_URL).mock(return_value=httpx.Response(200, json=data))
    (p,) = await FETCHERS["recruitee"](client, "acme")
    assert p.external_id == str(data["offers"][1]["id"]) and p.description_text


@respx.mock
async def test_recruitee_missing_offers_key_is_permanent(client):
    respx.get(REC_URL).mock(return_value=httpx.Response(200, json={"error": "nope"}))
    with pytest.raises(PermanentError):
        await FETCHERS["recruitee"](client, "acme")


# ---------- Workable (fixture is synthetic: see tests/fixtures/workable.json) ----------

WK_URL = "https://apply.workable.com/api/v1/widget/accounts/acme"


@respx.mock
async def test_workable_parses_jobs(client):
    route = respx.get(WK_URL).mock(return_value=httpx.Response(200, json=load("workable")))
    posts = await FETCHERS["workable"](client, "acme")
    assert route.calls[0].request.url.params["details"] == "true"
    p, empty = posts
    assert (p.provider, p.external_id) == ("workable", "A1B2C3D4E5")
    assert p.title == "Machine Learning Engineer" and p.url == "https://apply.workable.com/j/A1B2C3D4E5"
    assert p.location == "Dubai, United Arab Emirates"      # city == state collapses
    assert p.description_text == "We build LLM products & tooling.\n\n- Python\n- PyTorch"
    assert derive("workable", p.raw, p.location).remote_eligibility == "region_locked"
    assert empty.description_text == "" and empty.location == "Riyadh, Saudi Arabia"   # no state; ingest drops the empty one


@respx.mock
async def test_workable_429_surfaces_retry_after(client):
    respx.get(WK_URL).mock(return_value=httpx.Response(429, headers={"retry-after": "86280"}))
    with pytest.raises(RetryableError) as e:
        await FETCHERS["workable"](client, "acme")
    assert e.value.retry_after == 86280


@respx.mock
async def test_workable_unknown_account_is_permanent(client):
    respx.get(WK_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(PermanentError):
        await FETCHERS["workable"](client, "acme")


# ---------- Teamtailor (RSS only) ----------

TT_URL = "https://acme.teamtailor.com/jobs.rss"


@respx.mock
async def test_teamtailor_parses_rss(client):
    respx.get(TT_URL).mock(return_value=httpx.Response(200, text=(FIX / "teamtailor.rss").read_text()))
    posts = await FETCHERS["teamtailor"](client, "acme")
    assert len(posts) == 3
    p = posts[0]
    assert (p.provider, p.title) == ("teamtailor", "Customer Success Manager")
    assert p.external_id == "00156fd3-97f5-4b28-8ae7-bccb43179828"
    assert p.url == "https://careers.cleargrid.co/jobs/8589698-customer-success-manager"
    assert p.location == "Riyadh, Saudi Arabia"
    assert "About ClearGrid" in p.description_text and "<" not in p.description_text and "&lt;" not in p.description_text
    # derive() reads RSS child elements by tag name from raw
    assert p.raw["remoteStatus"] == "onsite" and p.raw["department"] == "Client Solutions"
    assert derive("teamtailor", p.raw, p.location).remote_eligibility == "onsite"
    assert derive("teamtailor", posts[1].raw, posts[1].location).remote_eligibility in ("eligible", "region_locked")


@respx.mock
async def test_teamtailor_multiple_locations_joined(client):
    respx.get(TT_URL).mock(return_value=httpx.Response(200, text=(FIX / "teamtailor.rss").read_text()))
    posts = await FETCHERS["teamtailor"](client, "acme")
    assert any(" | " in (p.location or "") for p in posts)


@respx.mock
async def test_teamtailor_custom_domain_slug_is_a_host(client):
    route = respx.get("https://careers.cleargrid.co/jobs.rss").mock(
        return_value=httpx.Response(200, text=(FIX / "teamtailor.rss").read_text()))
    assert len(await FETCHERS["teamtailor"](client, "careers.cleargrid.co")) == 3
    assert route.called


@respx.mock
async def test_teamtailor_not_rss_is_permanent(client):
    respx.get(TT_URL).mock(return_value=httpx.Response(200, text="<html>hello</html>"))
    with pytest.raises(PermanentError):
        await FETCHERS["teamtailor"](client, "acme")
    respx.get(TT_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(PermanentError):
        await FETCHERS["teamtailor"](client, "acme")


@respx.mock
async def test_teamtailor_rejects_entity_bombs(client):
    bomb = '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><rss><channel><title>&b;</title></channel></rss>'
    respx.get(TT_URL).mock(return_value=httpx.Response(200, text=bomb))
    with pytest.raises(PermanentError):
        await FETCHERS["teamtailor"](client, "acme")
