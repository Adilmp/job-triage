import csv

import pytest
from sqlalchemy import select

from triage import label_cli
from triage.models import Company, Label, Post, TriageRun


async def seed(session):
    au = Company(name="AuCo", region="AU")
    gulf = Company(name="GulfCo", region="GULF")
    bf = Company(name="BackCo", region="BACKFILL")
    session.add_all([au, gulf, bf])
    await session.flush()
    n = 0

    def mk(company, title, **kw):
        nonlocal n
        n += 1
        base = dict(company_id=company.id, provider="ashby", external_id=str(n), url=f"https://x/{n}", title=title,
                    location="Remote", description_text="Build things. " * 400, raw={}, fingerprint=f"f{n}", content_hash=f"h{n}",
                    remote_eligibility="eligible")
        return Post(**{**base, **kw})

    posts = [mk(bf, f"bf{i}") for i in range(4)] + [mk(au, "au1"), mk(au, "au2"), mk(gulf, "gulf1")]
    posts += [mk(au, "closed", closed_at=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc)),
              mk(au, "empty", description_text=""), mk(au, "dup", duplicate_of=1)]
    session.add_all(posts)
    await session.commit()
    return {p.title: p for p in posts}


@pytest.fixture
def fit_file(tmp_path, monkeypatch):
    f = tmp_path / "FIT.md"
    f.write_text("Fit = junior AI/backend engineer role I can do remotely from Pakistan.")
    monkeypatch.setattr(label_cli, "FIT_FILE", f)
    return f


async def titles(session, **kw):
    ids = await label_cli.unlabeled_ids(session, **kw)
    return [(await session.get(Post, i)).title for i in ids]


async def test_pool_is_au_and_gulf_first_and_excludes_closed_duplicate_empty_and_labeled(session):
    p = await seed(session)
    session.add(Label(post_id=p["au2"].id, source="gold", label="fit"))
    await session.commit()
    t = await titles(session, seed=1)
    assert set(t[:2]) == {"au1", "gulf1"} and set(t[2:]) == {"bf0", "bf1", "bf2", "bf3"}
    assert not {"closed", "empty", "dup", "au2"} & set(t)


async def test_order_is_seeded_and_region_filter_works(session):
    await seed(session)
    assert await titles(session, seed=7) == await titles(session, seed=7)
    others = [await titles(session, seed=a) for a in range(8, 14)]
    base = await titles(session, seed=7)
    assert any(o != base for o in others)
    assert set(await titles(session, region="GULF")) == {"gulf1"}


async def test_save_relabel_and_undo(session):
    p = await seed(session)
    pid = p["au1"].id
    await label_cli.save_label(session, pid, "fit", "remote + junior")
    await label_cli.save_label(session, pid, "not_fit", "changed my mind")      # relabel overwrites
    row = await session.get(Label, (pid, "gold"))
    assert (row.label, row.notes) == ("not_fit", "changed my mind")
    await label_cli.undo_label(session, pid)
    assert await session.get(Label, (pid, "gold")) is None


async def test_card_shows_only_title_company_location_url_and_1500_chars_and_no_model_output(session):
    p = await seed(session)
    session.add(TriageRun(post_id=p["au1"].id, post_version=1, criteria_version="v", jev_p=0.4242,
                          deep_read={"fit": True, "reasons": ["SECRETREASON"]}, route="deep_read"))
    await session.commit()
    card = await label_cli.card(session, p["au1"].id, "3/150")
    assert "au1" in card and "AuCo" in card and "Remote" in card and "https://x/" in card and "3/150" in card
    assert "Build things." in card and len(card) < 1500 + 400
    assert "0.4242" not in card and "SECRETREASON" not in card
    assert "jev" not in card.lower() and "fit?" in card.lower()


async def test_scripted_session_records_labels_notes_skip_undo_and_quit(session, fit_file):
    await seed(session)
    order = await label_cli.unlabeled_ids(session, seed=1)
    keys = iter(["f", "n", "s", "f", "u", "n", "q"])       # post0 f, post1 n, post2 skip, post3 f then undone,
    notes = iter(["remote junior", "", "will be undone", "second try"])   # post3 shown again -> n, then quit
    out = []
    await label_cli.run_session(session, read_key=lambda: next(keys), read_note=lambda: next(notes), write=out.append, seed=1)
    got = {l.post_id: (l.label, l.notes) for l in (await session.scalars(select(Label))).all()}
    assert got == {order[0]: ("fit", "remote junior"), order[1]: ("not_fit", None), order[3]: ("not_fit", "second try")}
    assert order[2] not in got                                  # skipped posts are left unlabeled
    assert out[-1] == "3 labeled"


async def test_undo_with_nothing_to_undo_just_warns(session, fit_file):
    await seed(session)
    keys = iter(["u", "q"])
    out = []
    await label_cli.run_session(session, read_key=lambda: next(keys), read_note=lambda: "", write=out.append, seed=1)
    assert "nothing to undo" in out


async def test_stops_at_the_limit(session, fit_file):
    await seed(session)
    out = []
    await label_cli.run_session(session, read_key=lambda: "f", read_note=lambda: "", write=out.append, limit=2, seed=1)
    assert out[-1] == "2 labeled" and await label_cli.gold_count(session) == 2


async def test_session_refuses_without_a_fit_definition(session, tmp_path, monkeypatch):
    monkeypatch.setattr(label_cli, "FIT_FILE", tmp_path / "missing.md")
    with pytest.raises(SystemExit, match="FIT"):
        await label_cli.run_session(session, read_key=lambda: "q", read_note=lambda: "", write=print)
    f = tmp_path / "todo.md"
    f.write_text("TODO: write one sentence")
    monkeypatch.setattr(label_cli, "FIT_FILE", f)
    with pytest.raises(SystemExit, match="FIT"):
        await label_cli.run_session(session, read_key=lambda: "q", read_note=lambda: "", write=print)


async def test_session_prints_the_fit_definition_and_progress(session, fit_file):
    await seed(session)
    out = []
    await label_cli.run_session(session, read_key=iter(["q"]).__next__, read_note=lambda: "", write=out.append, seed=1)
    text = "\n".join(out)
    assert "junior AI/backend engineer" in text and "0 labeled" in text


async def test_export_gold_csv(session, tmp_path):
    p = await seed(session)
    await label_cli.save_label(session, p["au1"].id, "fit", "good, remote")
    await label_cli.save_label(session, p["bf0"].id, "not_fit", None)
    session.add(Label(post_id=p["au2"].id, source="approval", label="fit"))      # approvals aren't gold
    await session.commit()
    path = tmp_path / "gold.csv"
    assert await label_cli.export_gold(session, path) == 2
    rows = list(csv.DictReader(path.open()))
    assert {(r["post_id"], r["label"], r["notes"]) for r in rows} == {
        (str(p["au1"].id), "fit", "good, remote"), (str(p["bf0"].id), "not_fit", "")}
    assert list(rows[0]) == ["post_id", "label", "notes", "tags", "provider", "external_id"]


async def test_stretch_key_and_hashtags_become_tags(session, fit_file):
    await seed(session)
    order = await label_cli.unlabeled_ids(session, seed=1)
    keys = iter(["t", "n", "f", "q"])
    notes = iter(["senior but asks 3 yrs", "pure research #research", "fine-tuning #finetuning #Research"])
    await label_cli.run_session(session, read_key=lambda: next(keys), read_note=lambda: next(notes), write=lambda *_: None, seed=1)
    got = {l.post_id: (l.label, l.notes, l.tags) for l in (await session.scalars(select(Label))).all()}
    assert got[order[0]] == ("stretch", "senior but asks 3 yrs", [])
    assert got[order[1]] == ("not_fit", "pure research #research", ["research"])
    assert got[order[2]] == ("fit", "fine-tuning #finetuning #Research", ["finetuning", "research"])


def test_parse_tags():
    assert label_cli.parse_tags("a #b c #d #B") == ["b", "d"]
    assert label_cli.parse_tags("") == [] and label_cli.parse_tags(None) == []
    assert label_cli.parse_tags("issue #12 and C# code") == ["12"]       # a '#' inside a word isn't a tag


async def test_pool_skips_posts_the_prefilter_would_drop(session):
    p = await seed(session)
    p["au1"].title = "Staff Engineer"
    p["au2"].remote_eligibility, p["au2"].location = "onsite", "London"
    p["gulf1"].title = "Senior ML Engineer"          # senior is allowed through now
    await session.commit()
    t = await titles(session, seed=1)
    assert "Staff Engineer" not in t and "au2" not in t and "Senior ML Engineer" in t
    everything = await titles(session, seed=1, include_prefiltered=True)
    assert {"Staff Engineer", "au2"} <= set(everything)


async def test_export_includes_tags(session, tmp_path):
    p = await seed(session)
    await label_cli.save_label(session, p["au1"].id, "stretch", "senior but ok #research")
    path = tmp_path / "gold.csv"
    await label_cli.export_gold(session, path)
    (row,) = list(csv.DictReader(path.open()))
    assert row["label"] == "stretch" and row["tags"] == "research"


async def test_review_lists_only_suspicious_fit_labels_and_lets_you_change_them(session, fit_file):
    p = await seed(session)
    p["au1"].title = "Staff Engineer"                                   # prefilter would drop it
    p["gulf1"].title = "Research Scientist"                             # research-titled
    p["bf0"].remote_eligibility = "region_locked"                       # US-only style remote
    await session.commit()
    for t in ("au1", "gulf1", "bf0", "bf1", "au2"):
        await label_cli.save_label(session, p[t].id, "fit", "too senior" if t == "bf1" else None)
    await label_cli.save_label(session, p["bf2"].id, "not_fit", None)
    p["bf3"].title = "Senior Data Engineer"                             # not_fit, senior, now passes the prefilter
    await session.commit()
    await label_cli.save_label(session, p["bf3"].id, "not_fit", None)
    ids = await label_cli.review_ids(session)
    assert {(await session.get(Post, i)).title for i in ids} == {"Staff Engineer", "Research Scientist", "bf0", "bf1", "Senior Data Engineer"}  # not au2, not bf2
    keys = iter(["n", "t", "k", "q"])
    notes = iter(["staff, no", "finetuning #finetuning"])
    out = []
    await label_cli.run_review(session, read_key=lambda: next(keys), read_note=lambda: next(notes), write=out.append)
    assert any("Current label: fit" in o for o in out)
    first, second = ids[0], ids[1]
    assert (await session.get(Label, (first, "gold"))).label == "not_fit"
    assert (await session.get(Label, (second, "gold"))).label == "stretch"
    assert (await session.get(Label, (ids[2], "gold"))).label == "not_fit"      # kept as it was


async def test_queue_shows_likely_relevant_posts_first_within_a_region(session):
    p = await seed(session)
    for t in ("au1", "au2", "gulf1", "closed", "empty", "dup"):
        p[t].closed_at = __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc)
    rows = {  # title: (location, derived status, description)
        "bf0": ("San Francisco", "unclear", "Come to our office."),                 # city only: tier 3
        "bf1": ("Remote - US", "region_locked", "US residents only."),              # region locked away from Pakistan: tier 4
        "bf2": ("Karachi, Pakistan", "hybrid", "Hybrid."),                          # Pakistan: tier 0
        "bf3": ("Remote", "eligible", "Anywhere."),                                 # eligible: tier 1
    }
    for t, (loc, status, desc) in rows.items():
        p[t].location, p[t].remote_eligibility, p[t].description_text = loc, status, desc
    extra = Post(company_id=p["bf0"].company_id, provider="ashby", external_id="x1", url="https://x/x1", title="bf4",
                 location="Berlin", remote_eligibility="unclear", description_text="Open to remote candidates in any timezone.",
                 raw={}, fingerprint="fx", content_hash="hx")
    apac = Post(company_id=p["bf0"].company_id, provider="ashby", external_id="x2", url="https://x/x2", title="bf5",
                location="Remote - APAC", remote_eligibility="region_locked", description_text="Work from your home.",
                raw={}, fingerprint="fy", content_hash="hy")
    session.add_all([extra, apac])
    await session.commit()
    order = await titles(session, seed=3)
    assert order[:2] == ["bf2", "bf3"] and set(order[2:4]) == {"bf4", "bf5"} and order[4:] == ["bf0", "bf1"]
