import logging
from datetime import timedelta

import pytest
from sqlalchemy import select

from triage.fetchers.base import RawPost
from triage.ingest import ingest
from triage.models import Company, Post


def post(ext="1", title="ML Engineer", location="Remote", desc="Build LLM things.", provider="ashby", raw=None):
    return RawPost(provider, ext, f"https://x/{ext}", title, location, desc, raw or {"workplaceType": "Remote"})


@pytest.fixture
async def acme(session):
    c = Company(name="Acme Pty Ltd", region="AU", board="ashby", slug="acme", verified=True)
    session.add(c)
    await session.commit()
    return c


async def posts_of(session):
    return (await session.scalars(select(Post).order_by(Post.id))).all()


async def test_new_post_is_inserted_derived_and_queued_for_triage(session, acme):
    res = await ingest(session, acme, [post()])
    await session.commit()
    (p,) = await posts_of(session)
    assert (p.version, p.status, p.closed_at, p.duplicate_of) == (1, "new", None, None)
    assert p.remote_eligibility == "eligible" and p.raw == {"workplaceType": "Remote"}
    assert len(p.content_hash) == len(p.fingerprint) == 64
    assert res.new == 1 and res.to_triage == [(p.id, 1)]
    assert acme.last_fetched_at is not None and acme.last_error is None


async def test_compensation_is_stored_with_currency(session, acme):
    raw = {"shouldDisplayCompensationOnJobPostings": True, "workplaceType": "Remote", "compensation": {"summaryComponents": [
        {"compensationType": "Salary", "interval": "1 YEAR", "currencyCode": "USD", "minValue": 100000, "maxValue": 120000}]}}
    await ingest(session, acme, [post(raw=raw)])
    await session.commit()
    (p,) = await posts_of(session)
    assert (p.comp_min, p.comp_max, p.comp_currency) == (100000, 120000, "USD")


async def test_unchanged_post_only_touches_last_seen(session, acme):
    await ingest(session, acme, [post()])
    await session.commit()
    (before,) = await posts_of(session)
    first_seen, last_seen = before.first_seen_at, before.last_seen_at
    res = await ingest(session, acme, [post()])
    await session.commit()
    await session.refresh(before)
    assert len(await posts_of(session)) == 1 and before.version == 1
    assert res.unchanged == 1 and res.new == 0 and res.to_triage == []
    assert before.first_seen_at == first_seen and before.last_seen_at > last_seen


async def test_edited_post_bumps_version_and_requeues_even_from_a_final_status(session, acme):
    await ingest(session, acme, [post()])
    await session.commit()
    (p,) = await posts_of(session)
    p.status = "rejected"
    await session.commit()
    res = await ingest(session, acme, [post(desc="Build LLM things. Now with agents.", raw={"workplaceType": "Hybrid"})])
    await session.commit()
    await session.refresh(p)
    assert (p.version, p.status) == (2, "new")
    assert "agents" in p.description_text and p.raw == {"workplaceType": "Hybrid"} and p.remote_eligibility == "hybrid"
    assert res.changed == 1 and res.to_triage == [(p.id, 2)]


async def test_missing_post_is_closed_and_reopens_when_it_returns(session, acme):
    await ingest(session, acme, [post("1"), post("2", title="Data Engineer")])
    await session.commit()
    res = await ingest(session, acme, [post("1")])
    await session.commit()
    p1, p2 = await posts_of(session)
    assert p1.closed_at is None and p2.closed_at is not None and res.closed == 1
    res = await ingest(session, acme, [post("1"), post("2", title="Data Engineer")])
    await session.commit()
    await session.refresh(p2)
    assert p2.closed_at is None and p2.version == 1 and res.to_triage == []


async def test_close_is_scoped_to_company_and_provider(session, acme):
    other = Company(name="Other", region="GULF", board="ashby", slug="other", verified=True)
    session.add(other)
    await session.commit()
    await ingest(session, other, [post("9", title="Other role")])
    await ingest(session, acme, [post("1", provider="ashby"), post("2", title="Data Engineer", provider="lever")])
    await session.commit()
    await ingest(session, acme, [post("1", provider="ashby")])        # the lever post wasn't part of this fetch
    await session.commit()
    assert all(p.closed_at is None for p in await posts_of(session))


async def test_cross_board_duplicate_is_linked_and_not_triaged(session, acme):
    r1 = await ingest(session, acme, [post("1", title="Sr. ML Eng", provider="greenhouse")])
    await session.commit()
    r2 = await ingest(session, acme, [post("abc", title="Senior ML Engineer", provider="lever")])
    await session.commit()
    first, second = await posts_of(session)
    assert second.duplicate_of == first.id and second.status == "duplicate"
    assert r1.to_triage == [(first.id, 1)] and r2.to_triage == [] and r2.duplicates == 1


async def test_duplicate_not_linked_to_a_closed_post_and_edits_of_duplicates_dont_requeue(session, acme):
    await ingest(session, acme, [post("1", provider="greenhouse")])
    await session.commit()
    await ingest(session, acme, [], provider="greenhouse")                # greenhouse post closes
    await ingest(session, acme, [post("abc", provider="lever")])
    await session.commit()
    first, second = await posts_of(session)
    assert first.closed_at is not None and second.duplicate_of is None and second.status == "new"
    await ingest(session, acme, [post("x", provider="ashby")])            # now a dup of the open lever post
    res = await ingest(session, acme, [post("x", provider="ashby", desc="changed")])
    await session.commit()
    assert res.changed == 1 and res.to_triage == []


async def test_same_provider_same_title_are_separate_posts(session, acme):
    # One req per city is normal (Ruya AI posts every role twice); only another *provider* makes a duplicate.
    res = await ingest(session, acme, [post("1", location="Dubai"), post("2", location="Serbia")])
    await session.commit()
    assert res.new == 2 and res.duplicates == 0 and all(p.duplicate_of is None for p in await posts_of(session))


async def test_empty_description_is_dropped_loudly_counted_and_not_closed(session, acme, caplog):
    await ingest(session, acme, [post("1")])
    await session.commit()
    with caplog.at_level(logging.WARNING, logger="triage.ingest"):
        res = await ingest(session, acme, [post("1", desc="  "), post("2", title="Empty", desc="")])
    await session.commit()
    assert res.dropped_empty == 2 and len(await posts_of(session)) == 1
    assert "2 posts dropped" in acme.last_error and "empty description" in acme.last_error
    assert any("empty description" in r.message for r in caplog.records)
    (p,) = await posts_of(session)
    assert p.closed_at is None and p.version == 1 and p.description_text == "Build LLM things."   # untouched
    await ingest(session, acme, [post("1")])
    assert acme.last_error is None                                         # cleared on a clean fetch


async def test_duplicate_primary_wins_by_lowest_id_and_dups_are_not_primaries(session, acme):
    await ingest(session, acme, [post("1", provider="greenhouse")])
    await ingest(session, acme, [post("2", provider="lever")])
    await ingest(session, acme, [post("3", provider="workable")])
    await session.commit()
    a, b, c = await posts_of(session)
    assert b.duplicate_of == a.id and c.duplicate_of == a.id
