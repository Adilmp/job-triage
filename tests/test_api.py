import asyncio

import httpx
import pytest
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from fakes import FakeRedis
from triage import api
from triage.config import settings
from triage.criteria import criteria_version
from triage.models import Company, Label, Post, TriageRun


@pytest.fixture
async def client(session):
    api.app.state.db = async_sessionmaker(session.bind, expire_on_commit=False)
    api.app.state.redis = FakeRedis()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t") as c:
        yield c


@pytest.fixture
async def data(session):
    au = Company(name="AuCo", region="AU", board="ashby", slug="au", verified=True)
    gulf = Company(name="GulfCo", region="GULF", board="ashby", slug="gulf", verified=True)
    session.add_all([au, gulf])
    await session.flush()

    def post(c, n, status="new", **kw):
        return Post(**{**dict(company_id=c.id, provider="ashby", external_id=str(n), url=f"https://x/{n}", title=f"Job {n}",
                              location="Remote", description_text="Build LLM things.", raw={}, fingerprint=f"f{n}",
                              content_hash=f"h{n}", remote_eligibility="eligible", status=status, version=1), **kw})
    posts = [post(au, 1, "awaiting_review"), post(au, 2, "shortlisted"), post(gulf, 3, "awaiting_review"),
             post(gulf, 4, "rejected"), post(au, 5)]
    session.add_all(posts)
    await session.flush()
    cv = criteria_version()
    session.add_all([
        TriageRun(post_id=posts[0].id, post_version=1, criteria_version=cv, jev_p=0.91, route="shortlist", latency_ms=100),
        TriageRun(post_id=posts[1].id, post_version=1, criteria_version=cv, jev_p=0.5, route="deep_read", latency_ms=300,
                  deep_read={"fit": True, "confidence": 0.8, "reasons": ["remote"], "red_flags": []},
                  final_status="shortlisted", decided_by="human", decision="approve"),
        TriageRun(post_id=posts[2].id, post_version=1, criteria_version=cv, jev_p=0.45, route="deep_read", latency_ms=200,
                  deep_read={"fit": False, "confidence": 0.4, "reasons": ["unclear"], "red_flags": ["US only?"]}),
        TriageRun(post_id=posts[3].id, post_version=1, criteria_version="old-criteria", jev_p=0.1, route="auto_reject",
                  latency_ms=50, final_status="rejected", decided_by="auto"),
    ])
    await session.commit()
    return {p.title: p.id for p in posts}


# ---------- decision (P9) ----------

async def test_decision_claims_writes_an_approval_label_and_queues_the_resume(client, data, session):
    pid = data["Job 1"]
    r = await client.post(f"/posts/{pid}/decision", json={"decision": "approve", "note": "apply today"})
    assert r.status_code == 200 and r.json()["status"] == "resuming"
    post = await session.get(Post, pid, populate_existing=True)
    assert post.status == "resuming"
    label = await session.get(Label, (pid, "approval"), populate_existing=True)
    assert (label.label, label.notes) == ("fit", "apply today")
    (name, args, job_id) = api.app.state.redis.jobs[0]
    assert (name, args) == ("resume_triage", (pid, "approve", "apply today")) and job_id.startswith(f"resume:{pid}:")


async def test_a_second_decision_gets_409(client, data):
    pid = data["Job 1"]
    assert (await client.post(f"/posts/{pid}/decision", json={"decision": "reject"})).status_code == 200
    r = await client.post(f"/posts/{pid}/decision", json={"decision": "approve"})
    assert r.status_code == 409 and "resuming" in r.json()["detail"]


async def test_two_simultaneous_decisions_one_wins(client, data):
    pid = data["Job 3"]
    a, b = await asyncio.gather(*(client.post(f"/posts/{pid}/decision", json={"decision": d}) for d in ("approve", "reject")))
    assert sorted((a.status_code, b.status_code)) == [200, 409]
    assert len(api.app.state.redis.jobs) == 1


async def test_reject_writes_a_not_fit_label(client, data, session):
    pid = data["Job 3"]
    await client.post(f"/posts/{pid}/decision", json={"decision": "reject"})
    assert (await session.get(Label, (pid, "approval"), populate_existing=True)).label == "not_fit"


async def test_decision_on_a_post_not_waiting_or_missing(client, data):
    assert (await client.post(f"/posts/{data['Job 2']}/decision", json={"decision": "approve"})).status_code == 409
    assert (await client.post("/posts/999999/decision", json={"decision": "approve"})).status_code == 404
    assert (await client.post(f"/posts/{data['Job 1']}/decision", json={"decision": "maybe"})).status_code == 422


async def test_if_the_queue_is_down_the_claim_is_undone(client, data, session, monkeypatch):
    async def down(*a, **kw):
        raise ConnectionError("redis down")
    monkeypatch.setattr(api.app.state.redis, "enqueue_job", down)
    pid = data["Job 1"]
    r = await client.post(f"/posts/{pid}/decision", json={"decision": "approve"})
    assert r.status_code == 503
    assert (await session.get(Post, pid, populate_existing=True)).status == "awaiting_review"
    assert await session.get(Label, (pid, "approval"), populate_existing=True) is None


# ---------- reading ----------

async def test_review_queue_lists_waiting_posts_with_score_and_reasons(client, data):
    r = await client.get("/posts", params={"status": "awaiting_review"})
    items = r.json()["items"]
    assert {i["title"] for i in items} == {"Job 1", "Job 3"}
    job3 = next(i for i in items if i["title"] == "Job 3")
    assert job3["jev_p"] == 0.45 and job3["route"] == "deep_read" and job3["deep_read"]["red_flags"] == ["US only?"]
    assert job3["company"] == "GulfCo" and job3["region"] == "GULF"
    assert "description_text" not in job3                                      # the list stays light


async def test_review_queue_filters_by_region_and_validates_status(client, data):
    r = await client.get("/posts", params={"status": "awaiting_review", "region": "AU"})
    assert [i["title"] for i in r.json()["items"]] == ["Job 1"]
    assert (await client.get("/posts", params={"status": "banana"})).status_code == 422


async def test_one_post_with_its_triage_history(client, data):
    r = await client.get(f"/posts/{data['Job 2']}")
    body = r.json()
    assert body["title"] == "Job 2" and body["description_text"] == "Build LLM things."
    (run,) = body["runs"]
    assert (run["route"], run["final_status"], run["decided_by"]) == ("deep_read", "shortlisted", "human")
    assert (await client.get("/posts/999999")).status_code == 404


async def test_stats(client, data):
    s = (await client.get("/stats")).json()
    assert s["posts_by_status"]["awaiting_review"] == 2 and s["posts_by_status"]["new"] == 1
    assert s["runs_by_route"] == {"shortlist": 1, "deep_read": 2, "auto_reject": 1}
    assert s["decided_by"] == {"human": 1, "auto": 1}
    assert s["latency_ms"]["p50"] == pytest.approx(150) and s["latency_ms"]["p95"] == pytest.approx(285)


# ---------- retriage (P7) ----------

async def test_retriage_stale_only_queues_posts_judged_under_old_criteria(client, data):
    r = await client.post("/retriage", params={"stale": "true"})
    assert r.json()["queued"] == 1
    (name, args, job_id) = api.app.state.redis.jobs[0]
    assert (name, args) == ("triage_post", (data["Job 4"], 1)) and criteria_version() in job_id


async def test_retriage_all_queues_never_triaged_and_stale_posts_up_to_the_limit(client, data):
    r = await client.post("/retriage", params={"stale": "false", "limit": 10})
    queued = {args[0] for _, args, _ in api.app.state.redis.jobs}
    assert r.json()["queued"] == 2 and queued == {data["Job 4"], data["Job 5"]}   # not the paused or finished ones
    api.app.state.redis.jobs.clear()
    assert (await client.post("/retriage", params={"stale": "false", "limit": 1})).json()["queued"] == 1
    assert (await client.post("/retriage", params={"limit": 100000})).status_code == 422          # no accidental mass run


# ---------- fetch trigger and health ----------

async def test_manual_fetch(client, data, session):
    cid = (await session.scalars(select(Company.id).where(Company.name == "AuCo"))).one()
    r = await client.post(f"/companies/{cid}/fetch")
    assert r.status_code == 202
    (name, args, _) = api.app.state.redis.jobs[0]
    assert (name, args) == ("fetch_company", (cid,))
    assert (await client.post("/companies/999999/fetch")).status_code == 404


@respx.mock
async def test_health(client, data):
    respx.route(host="testserver").pass_through()
    route = respx.get(f"{settings.OLLAMA_URL}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    r = await client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "db": "ok", "redis": "ok", "ollama": "ok"}
    route.mock(side_effect=httpx.ConnectError("down"))
    r = await client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["ollama"] == "down"
