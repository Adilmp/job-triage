import httpx
import pytest
import respx
from arq import Retry
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from triage import worker
from triage.fetchers.base import make_client
from triage.models import Company, Post

from fakes import FakeRedis

ASHBY = "https://api.ashbyhq.com/posting-api/job-board/acme"
JOB = {"id": "j1", "title": "ML Engineer", "jobUrl": "https://jobs.ashbyhq.com/acme/j1", "location": "Remote",
       "descriptionPlain": "Build LLM things.", "workplaceType": "Remote", "isRemote": True}


@pytest.fixture
async def ctx(session):
    redis, client = FakeRedis(), make_client()
    yield {"db": async_sessionmaker(session.bind, expire_on_commit=False), "http": client, "redis": redis, "job_try": 1}
    await client.aclose()


async def add_company(session, **kw):
    c = Company(**{"name": "Acme", "region": "AU", "board": "ashby", "slug": "acme", "verified": True, **kw})
    session.add(c)
    await session.commit()
    return c


async def company(session, cid):
    await session.rollback()
    return await session.get(Company, cid, populate_existing=True)


@respx.mock
async def test_fetch_company_fills_posts(session, ctx):
    c = await add_company(session)
    respx.get(ASHBY).mock(return_value=httpx.Response(200, json={"jobs": [JOB]}))
    out = await worker.fetch_company(ctx, c.id)
    assert out["new"] == 1
    (p,) = (await session.scalars(select(Post))).all()
    assert (p.title, p.status, p.remote_eligibility) == ("ML Engineer", "new", "eligible")
    assert (await company(session, c.id)).last_fetched_at is not None


@respx.mock
async def test_triage_jobs_use_idempotent_ids_and_have_an_off_switch(session, ctx, monkeypatch):
    c = await add_company(session)
    monkeypatch.setattr(worker.settings, "TRIAGE_ENABLED", False)
    respx.get(ASHBY).mock(return_value=httpx.Response(200, json={"jobs": [JOB]}))
    await worker.fetch_company(ctx, c.id)
    assert ctx["redis"].jobs == []                                    # off switch
    monkeypatch.setattr(worker.settings, "TRIAGE_ENABLED", True)
    respx.get(ASHBY).mock(return_value=httpx.Response(200, json={"jobs": [{**JOB, "descriptionPlain": "v2"}]}))
    await worker.fetch_company(ctx, c.id)
    pid = (await session.scalars(select(Post.id))).one()
    assert ctx["redis"].jobs == [("triage_post", (pid, 2), worker.triage_job_id(pid, 2))]
    assert worker.triage_job_id(pid, 2).startswith(f"triage:{pid}:2:")


@respx.mock
async def test_unverified_or_boardless_companies_are_skipped(session, ctx):
    a = await add_company(session, name="A", verified=False)
    b = await add_company(session, name="B", board=None, slug=None)
    route = respx.get(ASHBY).mock(return_value=httpx.Response(200, json={"jobs": []}))
    assert (await worker.fetch_company(ctx, a.id))["skipped"] and (await worker.fetch_company(ctx, b.id))["skipped"]
    assert not route.called


@respx.mock
async def test_503_retries_with_exponential_backoff_then_records_the_error(session, ctx):
    c = await add_company(session)
    respx.get(ASHBY).mock(return_value=httpx.Response(503))
    for try_no, base in ((1, 20), (3, 80), (4, 160)):
        ctx["job_try"] = try_no
        with pytest.raises(Retry) as e:
            await worker.fetch_company(ctx, c.id)
        assert base <= e.value.defer_score / 1000 <= base * 1.1 + 0.01
    ctx["job_try"] = worker.MAX_TRIES                                  # last try: give up, don't raise
    out = await worker.fetch_company(ctx, c.id)
    assert out["error"] and "503" in (await company(session, c.id)).last_error


@respx.mock
async def test_permanent_error_is_recorded_without_retry(session, ctx):
    c = await add_company(session)
    respx.get(ASHBY).mock(return_value=httpx.Response(404))
    out = await worker.fetch_company(ctx, c.id)                        # returns, no Retry
    assert out["error"] and "404" in (await company(session, c.id)).last_error


@respx.mock
async def test_long_retry_after_starts_a_site_cooldown_and_later_fetches_skip_the_site(session, ctx):
    c = await add_company(session, board="workable", slug="acme")
    d = await add_company(session, name="D", board="workable", slug="dco")
    route = respx.get("https://apply.workable.com/api/v1/widget/accounts/acme").mock(
        return_value=httpx.Response(429, headers={"retry-after": "86280"}))
    out = await worker.fetch_company(ctx, c.id)                        # 24h ask: no Retry job, just a recorded cooldown
    assert out["error"] and ctx["redis"].ttl["cooldown:workable.com"] == 86280
    assert "rate-limited" in (await company(session, c.id)).last_error
    other = respx.get("https://apply.workable.com/api/v1/widget/accounts/dco").mock(return_value=httpx.Response(200, json={"jobs": []}))
    out = await worker.fetch_company(ctx, d.id)                        # a different company on the same site
    assert out["skipped"] and not other.called and route.call_count == 1


@respx.mock
async def test_short_retry_after_is_honoured_as_the_retry_delay(session, ctx):
    c = await add_company(session)
    respx.get(ASHBY).mock(return_value=httpx.Response(429, headers={"retry-after": "300"}))
    with pytest.raises(Retry) as e:
        await worker.fetch_company(ctx, c.id)
    assert e.value.defer_score / 1000 >= 300


async def test_fetch_all_enqueues_verified_companies_once_per_window(session, ctx):
    a = await add_company(session, name="A")
    await add_company(session, name="B", verified=False)
    await add_company(session, name="C", board=None, slug=None, verified=False)
    await worker.fetch_all(ctx)
    await worker.fetch_all(ctx)                                        # overlapping cron in the same 6h window
    assert len(ctx["redis"].jobs) == 1
    name, args, job_id = ctx["redis"].jobs[0]
    assert (name, args) == ("fetch_company", (a.id,)) and job_id.startswith(f"fetch:{a.id}:")


def test_worker_settings():
    s = worker.WorkerSettings
    assert (s.max_tries, s.max_jobs, s.job_timeout) == (5, 4, 600)
    assert {f.__name__ for f in s.functions} == {"fetch_company", "fetch_all", "triage_post", "resume_triage", "expire_stale_reviews"}
    assert sorted(c.coroutine.__name__ for c in s.cron_jobs) == ["expire_stale_reviews", "fetch_all"]
