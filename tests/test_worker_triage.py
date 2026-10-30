import logging

import pytest
from arq import Retry
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from typing_extensions import TypedDict

from fakes import FakeRedis
from triage import graph as graph_mod
from triage import repo, worker
from triage.criteria import criteria_version
from triage.errors import PermanentError, RetryableError
from triage.models import Company, Label, Post, TriageRun
from triage.services import jev
from triage.services.deep_read import DeepRead


class Fakes:
    def __init__(self):
        self.jev_p, self.deep, self.jev_error = 0.95, DeepRead(fit=True, confidence=0.9), None
        self.jev_calls = self.deep_calls = 0


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()

    async def fake_ask(text, *a, **kw):
        f.jev_calls += 1
        if f.jev_error:
            raise f.jev_error
        return {"fit": f.jev_p}

    async def fake_deep_read(post):
        f.deep_calls += 1
        return f.deep

    monkeypatch.setattr(jev, "ask", fake_ask)
    monkeypatch.setattr(graph_mod, "deep_read", fake_deep_read)
    return f


@pytest.fixture
async def ctx(session, monkeypatch):
    monkeypatch.setattr(repo, "SessionLocal", async_sessionmaker(session.bind, expire_on_commit=False))
    saver = InMemorySaver()
    return {"db": async_sessionmaker(session.bind, expire_on_commit=False), "redis": FakeRedis(), "saver": saver,
            "graph": await graph_mod.build_graph(saver), "job_try": 1}


async def make_post(session, title="ML Engineer", **kw):
    c = (await session.scalars(select(Company))).first()
    if c is None:
        c = Company(name="Acme", region="AU", board="ashby", slug="acme", verified=True)
        session.add(c)
        await session.flush()
    n = len((await session.scalars(select(Post.id))).all()) + 1
    p = Post(**{**dict(company_id=c.id, provider="ashby", external_id=str(n), url=f"https://x/{n}", title=title, location="Remote",
                       description_text="Build LLM things.", raw={}, fingerprint=f"f{n}", content_hash=f"h{n}",
                       remote_eligibility="eligible", version=1), **kw})
    session.add(p)
    await session.commit()
    return p


async def fresh(session, p):
    await session.rollback()
    await session.refresh(p)
    return p


async def runs(session, p):
    pid = p if isinstance(p, int) else p.id     # rollback() expires p; read the id first
    await session.rollback()
    return (await session.scalars(select(TriageRun).where(TriageRun.post_id == pid).order_by(TriageRun.id))).all()


async def thread_exists(ctx, p):
    pid = p if isinstance(p, int) else p.id
    return bool((await ctx["graph"].aget_state({"configurable": {"thread_id": str(pid)}})).values)


# ---------- triage_post ----------

async def test_auto_reject_finishes_and_cleans_up_its_checkpoint(session, ctx, fakes):
    fakes.jev_p = 0.05
    p = await make_post(session)
    out = await worker.triage_post(ctx, p.id, 1)
    assert out == {"status": "rejected"} and (await fresh(session, p)).status == "rejected"
    assert not await thread_exists(ctx, p)                                   # P11


async def test_a_proposal_pauses_and_keeps_its_checkpoint(session, ctx, fakes):
    p = await make_post(session)
    assert await worker.triage_post(ctx, p.id, 1) == {"status": "awaiting_review"}
    assert (await fresh(session, p)).status == "awaiting_review" and await thread_exists(ctx, p)


async def test_running_the_same_version_twice_costs_nothing_and_adds_no_row(session, ctx, fakes):
    p = await make_post(session)
    await worker.triage_post(ctx, p.id, 1)
    assert await worker.triage_post(ctx, p.id, 1) == {"status": "awaiting_review"}
    assert fakes.jev_calls == 1 and len(await runs(session, p)) == 1


async def test_a_job_for_an_old_version_is_skipped(session, ctx, fakes):
    p = await make_post(session, version=3)
    assert await worker.triage_post(ctx, p.id, 2) == {"skipped": "stale version"}
    assert await worker.triage_post(ctx, 999999, 1) == {"skipped": "stale version"}
    assert fakes.jev_calls == 0


async def test_closed_and_duplicate_posts_are_not_triaged(session, ctx, fakes):
    from datetime import datetime, timezone
    closed = await make_post(session, closed_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    first = await make_post(session)
    dup = await make_post(session, duplicate_of=first.id, status="duplicate")
    assert (await worker.triage_post(ctx, closed.id, 1))["skipped"] == "closed"
    assert (await worker.triage_post(ctx, dup.id, 1))["skipped"] == "duplicate"
    assert fakes.jev_calls == 0


async def test_an_edited_post_replaces_the_old_pause_and_starts_over(session, ctx, fakes):
    p = await make_post(session)
    await worker.triage_post(ctx, p.id, 1)
    p.version, p.status = 2, "new"                                           # what ingest does on an edit
    await session.commit()
    assert await worker.triage_post(ctx, p.id, 2) == {"status": "awaiting_review"}
    assert fakes.jev_calls == 2
    assert [r.post_version for r in await runs(session, p)] == [1, 2]


async def test_a_criteria_change_restarts_a_paused_post(session, ctx, fakes, monkeypatch):
    p = await make_post(session)
    await worker.triage_post(ctx, p.id, 1)
    monkeypatch.setattr(worker.settings, "HI", 0.99)                         # new criteria_version
    await worker.triage_post(ctx, p.id, 1)
    assert fakes.jev_calls == 2 and len(await runs(session, p)) == 2


async def test_a_retryable_failure_retries_with_backoff_then_continues_from_the_checkpoint(session, ctx, fakes):
    p = await make_post(session)
    fakes.jev_error = RetryableError("Jev 503")
    with pytest.raises(Retry) as e:
        await worker.triage_post(ctx, p.id, 1)
    assert 20 <= e.value.defer_score / 1000 <= 22.1
    fakes.jev_error, ctx["job_try"] = None, 2
    assert await worker.triage_post(ctx, p.id, 1) == {"status": "awaiting_review"}
    assert (await fresh(session, p)).status == "awaiting_review"


async def test_after_the_last_try_the_error_lands_on_the_run_row(session, ctx, fakes):
    p = await make_post(session)
    fakes.jev_error, ctx["job_try"] = RetryableError("Jev 503"), worker.MAX_TRIES
    out = await worker.triage_post(ctx, p.id, 1)
    assert "error" in out
    (r,) = await runs(session, p)
    assert "Jev 503" in r.error and r.final_status is None and (await fresh(session, p)).status == "new"


async def test_a_permanent_failure_is_recorded_and_not_retried(session, ctx, fakes):
    p = await make_post(session)
    fakes.jev_error = PermanentError("Jev HTTP 401")
    out = await worker.triage_post(ctx, p.id, 1)                              # returns, no Retry
    assert "error" in out and fakes.jev_calls == 1
    (r,) = await runs(session, p)
    assert "401" in r.error


async def test_a_later_success_clears_the_recorded_error(session, ctx, fakes):
    p = await make_post(session)
    fakes.jev_error = PermanentError("Jev HTTP 401")
    await worker.triage_post(ctx, p.id, 1)
    fakes.jev_error, fakes.jev_p = None, 0.05
    await worker.triage_post(ctx, p.id, 1)
    (r,) = await runs(session, p)
    assert r.error is None and r.final_status == "rejected"


# ---------- resume_triage ----------

async def paused_post(session, ctx, fakes, **kw):
    p = await make_post(session, **kw)
    await worker.triage_post(ctx, p.id, 1)
    p.status = "resuming"                                                    # the API's claim (P9)
    await session.commit()
    return p


async def test_resume_approve_shortlists_and_cleans_up(session, ctx, fakes):
    p = await paused_post(session, ctx, fakes)
    out = await worker.resume_triage(ctx, p.id, "approve", "great")
    assert out == {"status": "shortlisted"} and (await fresh(session, p)).status == "shortlisted"
    assert not await thread_exists(ctx, p)
    (r,) = await runs(session, p)
    assert (r.final_status, r.decided_by, r.decision) == ("shortlisted", "human", "approve")
    assert fakes.jev_calls == 1


async def test_resume_reject_and_a_malformed_decision_both_reject(session, ctx, fakes):
    for decision in ("reject", "banana"):
        p = await paused_post(session, ctx, fakes)
        assert (await worker.resume_triage(ctx, p.id, decision))["status"] == "rejected"


async def test_nothing_pending_is_logged_and_the_post_goes_back_to_new(session, ctx, fakes, caplog):
    p = await make_post(session, status="resuming")                          # claimed, but its checkpoint is gone
    with caplog.at_level(logging.ERROR, logger="triage.worker"):
        out = await worker.resume_triage(ctx, p.id, "approve")
    assert "error" in out and (await fresh(session, p)).status == "new"
    assert any("no pending approval" in r.message for r in caplog.records)


async def test_an_unknown_interrupt_kind_is_an_error_not_a_silent_resume(session, ctx, fakes, caplog):
    class S(TypedDict, total=False):
        x: int

    async def weird(state):
        interrupt({"kind": "something_else"})
        return {"x": 1}

    g = StateGraph(S)
    g.add_node("weird", weird)
    g.add_edge(START, "weird")
    g.add_edge("weird", END)
    ctx["graph"] = g.compile(checkpointer=ctx["saver"])
    p = await make_post(session, status="resuming")
    await ctx["graph"].ainvoke({}, {"configurable": {"thread_id": str(p.id)}})
    with caplog.at_level(logging.ERROR, logger="triage.worker"), pytest.raises(ValueError, match="something_else"):
        await worker.resume_triage(ctx, p.id, "approve")
    assert (await fresh(session, p)).status == "awaiting_review"            # not stuck in 'resuming'
    assert any("unknown interrupt kind" in r.message for r in caplog.records)


async def test_resume_failure_retries_then_puts_the_post_back_for_review(session, ctx, fakes, monkeypatch):
    p = await paused_post(session, ctx, fakes)

    async def boom(*a, **kw):
        raise RetryableError("db blip")

    real = ctx["graph"].ainvoke
    monkeypatch.setattr(ctx["graph"], "ainvoke", boom)
    with pytest.raises(Retry):
        await worker.resume_triage(ctx, p.id, "approve")
    ctx["job_try"] = worker.MAX_TRIES
    out = await worker.resume_triage(ctx, p.id, "approve")
    assert "error" in out and (await fresh(session, p)).status == "awaiting_review"
    monkeypatch.setattr(ctx["graph"], "ainvoke", real)
    assert (await worker.resume_triage(ctx, p.id, "approve"))["status"] == "shortlisted"      # and it can still be decided


# ---------- expire_stale_reviews (P12) ----------

async def age_run(session, p, days):
    await session.execute(text("update triage_runs set created_at = now() - make_interval(days => :d) where post_id = :p"),
                          {"d": days, "p": p.id})
    await session.commit()


async def test_old_reviews_expire_without_a_label_and_fresh_ones_stay(session, ctx, fakes):
    old, new, claimed = [await make_post(session) for _ in range(3)]
    old_id, new_id = old.id, new.id
    for p in (old, new, claimed):
        await worker.triage_post(ctx, p.id, 1)
    claimed.status = "resuming"
    await session.commit()
    await age_run(session, old, 8)
    await age_run(session, claimed, 8)
    assert await worker.expire_stale_reviews(ctx) == 1
    assert (await fresh(session, old)).status == "expired" and (await fresh(session, new)).status == "awaiting_review"
    assert (await fresh(session, claimed)).status == "resuming"              # someone is deciding right now: leave it
    assert not await thread_exists(ctx, old_id) and await thread_exists(ctx, new_id)
    assert (await runs(session, old_id))[0].final_status == "expired"
    assert (await session.scalars(select(Label))).all() == []               # a timeout is not a "no": no label (P12)
    assert await worker.expire_stale_reviews(ctx) == 0
