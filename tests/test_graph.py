import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from triage import graph, repo
from triage.config import settings
from triage.models import Company, Post, TriageRun
from triage.services import jev
from triage.services.deep_read import DeepRead


class Fakes:
    def __init__(self):
        self.jev_p, self.deep = 0.5, DeepRead(fit=True, confidence=0.9, reasons=["remote"], red_flags=[])
        self.jev_calls = self.deep_calls = 0


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()

    async def fake_ask(text, *a, **kw):
        f.jev_calls += 1
        return {"fit": f.jev_p}

    async def fake_deep_read(post):
        f.deep_calls += 1
        return f.deep

    monkeypatch.setattr(jev, "ask", fake_ask)
    monkeypatch.setattr(graph, "deep_read", fake_deep_read)
    return f


@pytest.fixture
async def db(session, monkeypatch):
    monkeypatch.setattr(repo, "SessionLocal", async_sessionmaker(session.bind, expire_on_commit=False))
    c = Company(name="Acme", region="AU", board="ashby", slug="acme", verified=True)
    session.add(c)
    await session.commit()
    return session, c


async def make_post(db, title="ML Engineer", location="Remote", remote="eligible", n=[0]):
    session, c = db
    n[0] += 1
    p = Post(company_id=c.id, provider="ashby", external_id=str(n[0]), url=f"https://x/{n[0]}", title=title, location=location,
             description_text="Build LLM things.", raw={}, fingerprint=f"f{n[0]}", content_hash=f"h{n[0]}",
             remote_eligibility=remote, version=1)
    session.add(p)
    await session.commit()
    return p


async def run(p, saver=None):
    g = await graph.build_graph(saver or InMemorySaver())
    cfg = {"configurable": {"thread_id": str(p.id)}}
    return g, cfg, await g.ainvoke({"post_id": p.id}, cfg)


async def row(session, p):
    await session.rollback()
    await session.refresh(p)
    run_row = (await session.scalars(select(TriageRun).where(TriageRun.post_id == p.id))).one()
    return p.status, run_row


def paused(out):
    return out.get("__interrupt__")


# ---------- pure routing ----------

def test_screen_route_boundaries():
    assert graph.screen_route(0.7, 0.7, 0.2) == "shortlist"          # >= HI
    assert graph.screen_route(0.2, 0.7, 0.2) == "auto_reject"        # <= LO
    assert graph.screen_route(0.45, 0.7, 0.2) == "deep_read"
    assert graph.screen_route(0.6999, 0.7, 0.2) == "deep_read" and graph.screen_route(0.2001, 0.7, 0.2) == "deep_read"


def test_deep_read_sends_fits_and_unsure_rejections_to_the_human():
    sure_no = DeepRead(fit=False, confidence=0.9)
    assert graph.needs_human(DeepRead(fit=True, confidence=0.3), 0.7) == (True, "deep_read_fit")
    assert graph.needs_human(sure_no, 0.7) == (False, "deep_read_reject")
    assert graph.needs_human(DeepRead(fit=False, confidence=0.4), 0.7) == (True, "deep_read_unsure_reject")
    assert graph.needs_human(DeepRead(fit=False, confidence=0.7), 0.7) == (False, "deep_read_reject")     # at the line: confident enough


@pytest.mark.parametrize("answer,want", [
    ("approve", "approve"), ("APPROVE ", "approve"), ({"decision": "approve", "note": "yes!"}, "approve"),
    ("reject", "reject"), ({"decision": "reject"}, "reject"),
    (None, "reject"), ("banana", "reject"), ({}, "reject"), ({"decision": 42}, "reject"), (["approve"], "reject"), (True, "reject"),
])
def test_resume_values_parse_and_garbage_means_reject(answer, want):
    assert graph.parse_decision(answer)[0] == want


# ---------- whole graph ----------

async def test_prefilter_drop_ends_without_calling_any_model(db, fakes):
    p = await make_post(db, title="Staff Software Engineer")
    _, _, out = await run(p)
    assert not paused(out) and fakes.jev_calls == fakes.deep_calls == 0
    status, r = await row(db[0], p)
    assert (status, r.route, r.final_status, r.decided_by) == ("rejected", "prefilter_drop", "rejected", "auto")
    assert r.jev_p is None and r.latency_ms >= 0 and "seniority" in r.reason and r.error is None


async def test_low_score_is_auto_rejected_without_deep_read(db, fakes):
    fakes.jev_p = 0.05
    p = await make_post(db)
    _, _, out = await run(p)
    assert not paused(out) and fakes.deep_calls == 0
    status, r = await row(db[0], p)
    assert (status, r.route, r.jev_p, r.decided_by) == ("rejected", "auto_reject", 0.05, "auto")
    assert (r.hi, r.lo) == (settings.HI, settings.LO) and r.criteria_version


async def test_high_score_pauses_for_approval_then_resumes_to_shortlisted(db, fakes):
    fakes.jev_p = 0.95
    p = await make_post(db)
    g, cfg, out = await run(p)
    (it,) = paused(out)
    assert it.value["kind"] == "approval_request" and it.value["post_id"] == p.id
    assert it.value["title"] == "ML Engineer" and it.value["jev_p"] == 0.95 and it.value["why"] == "jev_high"
    status, r = await row(db[0], p)
    assert status == "awaiting_review" and r.route == "shortlist" and r.final_status is None     # visible in the review queue
    out = await g.ainvoke(Command(resume={"decision": "approve", "note": "apply"}), cfg)
    assert not paused(out) and out["status"] == "shortlisted" and out["decision"] == "approve"
    status, r = await row(db[0], p)
    assert (status, r.final_status, r.decided_by, r.decision) == ("shortlisted", "shortlisted", "human", "approve")
    assert fakes.jev_calls == 1 and fakes.deep_calls == 0           # the resume did not call Jev again


async def test_reject_and_garbage_resume_both_reject(db, fakes):
    fakes.jev_p = 0.95
    for answer in ("reject", {"nonsense": 1}):
        p = await make_post(db)
        g, cfg, _ = await run(p)
        out = await g.ainvoke(Command(resume=answer), cfg)
        assert out["status"] == "rejected" and out["decision"] == "reject"
        status, r = await row(db[0], p)
        assert (status, r.decided_by) == ("rejected", "human")


async def test_middle_score_goes_to_deep_read_and_a_fit_reaches_the_human(db, fakes):
    fakes.jev_p = 0.45
    p = await make_post(db)
    g, cfg, out = await run(p)
    (it,) = paused(out)
    assert fakes.deep_calls == 1 and it.value["why"] == "deep_read_fit" and it.value["deep_read"]["reasons"] == ["remote"]
    await g.ainvoke(Command(resume="approve"), cfg)
    status, r = await row(db[0], p)
    assert status == "shortlisted" and r.route == "deep_read" and r.deep_read["fit"] is True
    assert fakes.deep_calls == 1                                    # still one after the resume


async def test_confident_deep_read_rejection_is_final_and_needs_no_human(db, fakes):
    fakes.jev_p, fakes.deep = 0.45, DeepRead(fit=False, confidence=0.95, red_flags=["US only"])
    p = await make_post(db)
    _, _, out = await run(p)
    assert not paused(out)
    status, r = await row(db[0], p)
    assert (status, r.decided_by, r.route) == ("rejected", "auto", "deep_read") and r.deep_read["red_flags"] == ["US only"]


async def test_unsure_deep_read_rejection_is_sent_to_the_human(db, fakes):
    fakes.jev_p, fakes.deep = 0.45, DeepRead(fit=False, confidence=0.4, reasons=["maybe remote?"])
    p = await make_post(db)
    g, cfg, out = await run(p)
    (it,) = paused(out)
    assert it.value["why"] == "deep_read_unsure_reject"
    await g.ainvoke(Command(resume="approve"), cfg)
    assert (await row(db[0], p))[0] == "shortlisted"


async def test_thresholds_are_copied_into_state_and_stay_put(db, fakes, monkeypatch):
    fakes.jev_p = 0.95
    p = await make_post(db)
    g, cfg, _ = await run(p)
    monkeypatch.setattr(settings, "HI", 0.99)
    monkeypatch.setattr(settings, "LO", 0.9)
    monkeypatch.setattr(settings, "DEEP_READ_MIN_CONFIDENCE", 0.1)
    out = await g.ainvoke(Command(resume="approve"), cfg)
    assert (out["hi"], out["lo"], out["min_conf"]) == (0.7, 0.2, 0.7)
    status, r = await row(db[0], p)
    assert (r.hi, r.lo, status) == (0.7, 0.2, "shortlisted")


async def test_state_holds_only_plain_values(db, fakes):
    fakes.jev_p = 0.45
    p = await make_post(db)
    g, cfg, _ = await run(p)
    state = (await g.aget_state(cfg)).values

    def plain(v):
        return isinstance(v, (str, int, float, bool, type(None))) or (
            isinstance(v, dict) and all(plain(x) for x in v.values())) or (isinstance(v, list) and all(plain(x) for x in v))
    assert all(plain(v) for v in state.values()), state


async def test_post_and_criteria_version_are_loaded_when_not_passed(db, fakes):
    fakes.jev_p = 0.05
    p = await make_post(db)
    _, _, out = await run(p)
    assert out["post"]["title"] == "ML Engineer" and len(out["criteria_version"]) == 16


async def test_a_post_that_does_not_exist_is_an_error(db, fakes):
    g = await graph.build_graph(InMemorySaver())
    with pytest.raises(ValueError, match="not found"):
        await g.ainvoke({"post_id": 999999}, {"configurable": {"thread_id": "x"}})


async def test_a_rerun_node_does_not_duplicate_the_run_row(db, fakes):
    fakes.jev_p = 0.95
    p = await make_post(db)
    saver = InMemorySaver()
    g, cfg, _ = await run(p, saver)
    g2 = await graph.build_graph(saver)                              # a fresh process, same checkpoint store
    await g2.ainvoke(Command(resume="approve"), cfg)
    session = db[0]
    await session.rollback()
    assert len((await session.scalars(select(TriageRun).where(TriageRun.post_id == p.id))).all()) == 1


async def test_pause_survives_a_restart_with_the_postgres_checkpointer(db, fakes, pg_url):
    """The real thing: pause, drop every in-memory object, resume from the checkpoint tables."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    url = pg_url.replace("+asyncpg", "")
    fakes.jev_p = 0.45
    p = await make_post(db)
    cfg = {"configurable": {"thread_id": str(p.id)}}

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        out = await (await graph.build_graph(saver)).ainvoke({"post_id": p.id}, cfg)
        assert paused(out) and fakes.jev_calls == 1 and fakes.deep_calls == 1

    fakes.jev_p = 0.0                                               # "restart": a new saver, a new graph, a changed world
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        g = await graph.build_graph(saver)
        assert (await g.aget_state(cfg)).values["jev_p"] == 0.45     # state came back from Postgres
        out = await g.ainvoke(Command(resume="approve"), cfg)
        assert out["status"] == "shortlisted" and fakes.jev_calls == 1 and fakes.deep_calls == 1
        await saver.adelete_thread(str(p.id))                        # what the worker does after a decision
        assert (await g.aget_state(cfg)).values == {}
    assert (await row(db[0], p))[0] == "shortlisted"
