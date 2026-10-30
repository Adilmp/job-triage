import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from triage import repo
from triage.models import Company, Post, TriageRun


@pytest.fixture
async def db(session, monkeypatch):
    monkeypatch.setattr(repo, "SessionLocal", async_sessionmaker(session.bind, expire_on_commit=False))
    c = Company(name="Acme", region="AU", board="ashby", slug="acme", verified=True)
    session.add(c)
    await session.flush()
    p = Post(company_id=c.id, provider="ashby", external_id="1", url="https://x/1", title="ML Engineer", location="Remote",
             description_text="Build things.", raw={}, fingerprint="f", content_hash="h", remote_eligibility="eligible", version=2)
    session.add(p)
    await session.commit()
    return p


async def runs(session):
    await session.rollback()
    return (await session.scalars(select(TriageRun).order_by(TriageRun.id))).all()


async def test_get_post_dict_has_everything_the_graph_needs(db):
    d = await repo.get_post_dict(db.id)
    assert d["id"] == db.id and d["version"] == 2 and d["company"] == "Acme"
    assert {"title", "location", "url", "description_text", "remote_eligibility"} <= set(d)
    assert all(isinstance(v, (str, int, float, type(None))) for v in d.values())     # plain values: the checkpointer can store them
    assert await repo.get_post_dict(999999) is None


async def test_set_status(db, session):
    await repo.set_status(db.id, "awaiting_review")
    await session.refresh(db)
    assert db.status == "awaiting_review"
    with pytest.raises(ValueError, match="status"):
        await repo.set_status(db.id, "banana")


async def test_record_run_inserts_then_updates_the_same_row_when_a_node_reruns(db, session):
    kw = dict(post_id=db.id, post_version=2, criteria_version="abc")
    first = await repo.record_run(**kw, jev_p=0.5, hi=0.7, lo=0.2, route="deep_read", deep_read={"fit": True})
    second = await repo.record_run(**kw, jev_p=0.5, hi=0.7, lo=0.2, route="deep_read", deep_read={"fit": True},
                                   final_status="shortlisted", decided_by="human", decision="approve", latency_ms=1200)
    assert first == second
    (r,) = await runs(session)
    assert (r.jev_p, r.route, r.final_status, r.decided_by, r.decision, r.latency_ms) == (0.5, "deep_read", "shortlisted", "human", "approve", 1200)
    assert r.deep_read == {"fit": True}                              # earlier fields survive a later partial update


async def test_a_new_criteria_version_or_post_version_is_a_new_row(db, session):
    await repo.record_run(post_id=db.id, post_version=2, criteria_version="a", route="auto_reject")
    await repo.record_run(post_id=db.id, post_version=2, criteria_version="b", route="auto_reject")
    await repo.record_run(post_id=db.id, post_version=3, criteria_version="a", route="auto_reject")
    assert len(await runs(session)) == 3


async def test_record_run_validates_decided_by(db):
    with pytest.raises(Exception):
        await repo.record_run(post_id=db.id, post_version=2, criteria_version="a", decided_by="robot")
