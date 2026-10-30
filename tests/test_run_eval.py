import csv

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

import run_eval as ev
from triage import graph as graph_mod
from triage import repo
from triage.config import settings
from triage.models import Company, Post, TriageRun
from triage.services import jev
from triage.services.deep_read import DeepRead


# ---------- metrics ----------

def test_wilson_known_values():
    w = ev.wilson(0, 10)
    assert w["p"] == 0 and w["lo"] == 0 and w["hi"] == pytest.approx(0.2775, abs=1e-3)
    w = ev.wilson(10, 10)
    assert w["p"] == 1 and w["hi"] == 1 and w["lo"] == pytest.approx(0.7225, abs=1e-3)
    w = ev.wilson(5, 10)
    assert (w["lo"], w["hi"]) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert ev.wilson(0, 0)["p"] is None


def test_auc():
    assert ev.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert ev.auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert ev.auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5          # ties count half
    assert ev.auc([0.9, 0.4, 0.6, 0.1], [True, True, False, False]) == 0.75
    assert ev.auc([0.9, 0.8], [True, True]) is None


def test_pick_thresholds():
    scores = [0.02, 0.05, 0.10, 0.30, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95]
    pos = [False, False, False, True, False, False, True, True, True, False]
    t = ev.pick_thresholds(scores, pos, min_precision=0.75, min_support=3)
    assert t["lo"] == 0.10          # highest score below every positive
    assert t["hi"] == 0.70          # 0.70+: 3 of 4 positive
    assert ev.pick_thresholds([0.1, 0.2], [False, False])["hi"] == 1.0              # no positives: nothing to shortlist


def test_threshold_table_counts():
    rows = {r["t"]: r for r in ev.threshold_table([0.1, 0.5, 0.9], [False, True, True], steps=[0.5])}
    assert rows[0.5] == {"t": 0.5, "n_above": 2, "precision_above": 1.0, "recall_above": 1.0, "positives_at_or_below": 1}


def rec(pid, positive, route, jev_p, proposed, **kw):
    return {"post_id": pid, "positive": positive, "label": "fit" if positive else "not_fit", "tags": [], "route": route,
            "jev_p": jev_p, "proposed": proposed, "title": f"t{pid}", "company": "c", "model_ms": 100.0 * pid,
            "cost_usd": 0.00002, **kw}


def test_report_numbers():
    records = [
        rec(1, True, "shortlist", 0.9, True),
        rec(2, True, "auto_reject", 0.1, False),                  # false reject at Jev
        rec(3, True, "prefilter_drop", None, False, reason="seniority"),   # false reject at prefilter
        rec(4, False, "deep_read", 0.5, True, deep_read={"fit": True}),    # false alarm
        rec(5, True, "deep_read", 0.4, True, deep_read={"fit": True}),
        rec(6, False, "auto_reject", 0.05, False),
    ]
    r = ev.report(records, hi=0.7, lo=0.2)
    assert (r["n"], r["positives"]) == (6, 4)
    assert r["false_reject"]["k"] == 2 and r["false_reject"]["n"] == 4
    assert r["false_reject"]["by_stage"] == {"prefilter": 1, "jev_low": 1, "deep_read": 0}
    assert r["end_to_end"]["precision"]["k"] == 2 and r["end_to_end"]["precision"]["n"] == 3
    assert r["end_to_end"]["recall"]["p"] == 0.5 and r["end_to_end"]["review_per_100"] == 50
    assert r["router"]["prefilter_drop"]["k"] == 1 and r["deep_read"]["n"] == 2
    assert r["deep_read"]["pipeline_accuracy"]["k"] == 1                      # post 5 right, post 4 wrong
    assert r["jev_alone"]["auc"] == pytest.approx(4 / 6)                  # 4 of 6 positive-negative pairs ordered right
    assert [f["post_id"] for f in r["failures"]["false_rejects"]] == [2, 3]
    assert r["failures"]["false_rejects"][1]["prefilter_reason"] == "seniority"
    assert [f["post_id"] for f in r["failures"]["false_alarms"]] == [4]
    assert r["end_to_end"]["jev_cost_usd"] == pytest.approx(6 * 0.00002)


def test_gate():
    base = {"false_reject": ev.wilson(2, 40), "end_to_end": {"recall": ev.wilson(30, 40)}}
    ok = {"false_reject": ev.wilson(3, 40), "end_to_end": {"recall": ev.wilson(28, 40)}}
    bad = {"false_reject": ev.wilson(12, 40), "end_to_end": {"recall": ev.wilson(15, 40)}}
    assert ev.gate(ok, base) == [] and ev.gate(ok, None) == []
    problems = ev.gate(bad, base)
    assert len(problems) == 2 and "false-reject" in problems[0] and "recall" in problems[1]


# ---------- split ----------

def test_split_is_stratified_seeded_and_frozen(tmp_path):
    gold = [{"post_id": i, "label": lab, "positive": lab != "not_fit", "tags": []}
            for i, lab in enumerate(["fit"] * 20 + ["stretch"] * 30 + ["not_fit"] * 200)]
    s = ev.make_split(gold)
    assert s == ev.make_split(gold)                                         # same seed, same split
    held = [g for g in gold if s[g["post_id"]] == "heldout"]
    assert len(held) == 100 and sum(g["label"] == "fit" for g in held) == 8 and sum(g["label"] == "stretch" for g in held) == 12
    path = tmp_path / "split.csv"
    ev.write_split(s, path)
    assert ev.read_split(path) == s
    with pytest.raises(SystemExit, match="frozen"):
        ev.write_split(s, path)


def test_load_gold_stretch_switch(tmp_path):
    path = tmp_path / "gold.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id", "label", "notes", "tags", "provider", "external_id"])
        w.writerows([[1, "fit", "", "", "a", "1"], [2, "stretch", "", "research", "a", "2"], [3, "not_fit", "", "", "a", "3"]])
    assert [g["positive"] for g in ev.load_gold(path)] == [True, True, False]
    assert [g["positive"] for g in ev.load_gold(path, stretch_positive=False)] == [True, False, False]
    assert ev.load_gold(path)[1]["tags"] == ["research"]


# ---------- the whole run, against the real graph ----------

@pytest.fixture
async def posts(session, monkeypatch):
    monkeypatch.setattr(repo, "SessionLocal", async_sessionmaker(session.bind, expire_on_commit=False))
    c = Company(name="Acme", region="AU", board="ashby", slug="acme", verified=True)
    session.add(c)
    await session.flush()
    titles = {"high": "ML Engineer", "low": "Data Engineer", "mid": "AI Engineer", "staff": "Staff Engineer"}
    rows = {}
    for i, (k, t) in enumerate(titles.items()):
        p = Post(company_id=c.id, provider="ashby", external_id=str(i), url=f"https://x/{i}", title=t, location="Remote",
                 description_text=f"post {k}", raw={}, fingerprint=f"f{i}", content_hash=f"h{i}", remote_eligibility="eligible")
        session.add(p)
        rows[k] = p
    await session.commit()
    return {k: p.id for k, p in rows.items()}


@pytest.fixture
def models(monkeypatch):
    calls = {"jev": 0, "deep": 0}
    scores = {"post high": 0.95, "post low": 0.05, "post mid": 0.45}

    async def fake_ask_with_usage(text, questions=None, client=None):
        calls["jev"] += 1
        return {"fit": next(v for k, v in scores.items() if k in text)}, {"input_tokens": 500, "model": "jev-test"}

    async def fake_deep_read(post):
        calls["deep"] += 1
        return DeepRead(fit=False, confidence=0.9, red_flags=["US only"])

    monkeypatch.setattr(jev, "ask_with_usage", fake_ask_with_usage)
    monkeypatch.setattr(graph_mod, "deep_read", fake_deep_read)
    return calls


async def test_evaluate_runs_the_graph_answers_pauses_from_gold_and_writes_nothing(session, posts, models, tmp_path):
    gold = [{"post_id": posts["high"], "label": "fit", "positive": True, "tags": []},
            {"post_id": posts["low"], "label": "not_fit", "positive": False, "tags": []},
            {"post_id": posts["mid"], "label": "fit", "positive": True, "tags": []},
            {"post_id": posts["staff"], "label": "stretch", "positive": True, "tags": []}]
    real = (jev.ask, graph_mod.deep_read, repo.set_status, repo.record_run, settings.HI, settings.LO)
    cache = ev.AnswerCache(tmp_path / "cache.json")
    records, meta = await ev.evaluate(gold, hi=0.7, lo=0.2, cache=cache, progress=lambda *_: None)
    by = {r["post_id"]: r for r in records}

    assert by[posts["high"]]["route"] == "shortlist" and by[posts["high"]]["final_status"] == "shortlisted"   # gold said approve
    assert by[posts["low"]]["route"] == "auto_reject"
    assert by[posts["mid"]]["route"] == "deep_read" and not by[posts["mid"]]["proposed"]   # a confident "no": a false reject
    assert by[posts["staff"]]["route"] == "prefilter_drop" and by[posts["staff"]]["jev_p"] is None
    assert by[posts["high"]]["cost_usd"] == pytest.approx(jev.cost_usd({"input_tokens": 500}))
    assert meta["fresh_jev_calls"] == 3 and meta["fresh_deep_reads"] == 1

    # nothing was written to the database, and every swapped function and setting is back
    await session.rollback()
    assert await session.scalar(select(func.count()).select_from(TriageRun)) == 0
    assert set((await session.scalars(select(Post.status))).all()) == {"new"}
    assert (jev.ask, graph_mod.deep_read, repo.set_status, repo.record_run, settings.HI, settings.LO) == real

    r = ev.report(records, 0.7, 0.2)
    assert r["false_reject"]["by_stage"] == {"prefilter": 1, "jev_low": 0, "deep_read": 1}

    # a second run is served from the cache: no new model calls, same answers, same reported cost
    records2, meta2 = await ev.evaluate(gold, hi=0.7, lo=0.2, cache=ev.AnswerCache(tmp_path / "cache.json"), progress=lambda *_: None)
    assert models == {"jev": 3, "deep": 1} and meta2["fresh_jev_calls"] == 0
    assert sorted((r["post_id"], r["route"], r["cost_usd"]) for r in records2) == sorted((r["post_id"], r["route"], r["cost_usd"]) for r in records)


async def test_thresholds_from_the_command_line_change_the_routing(session, posts, models, tmp_path):
    gold = [{"post_id": posts["mid"], "label": "fit", "positive": True, "tags": []}]
    records, meta = await ev.evaluate(gold, hi=0.4, lo=0.1, cache=ev.AnswerCache(tmp_path / "c.json"), progress=lambda *_: None)
    assert records[0]["route"] == "shortlist" and meta["hi"] == 0.4


def test_lo_options_trade_misses_for_deep_reads():
    scores = [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.5, 0.9]
    pos = [False, True, False, True, False, True, False, True, True]
    opts = ev.lo_options(scores, pos, hi=0.45, max_misses=2)
    assert [o["misses"] for o in opts] == [0, 1, 2] and len(opts) == len({o["lo"] for o in opts})
    assert opts[0]["lo"] < 0.01 or opts[0]["lo"] == 0.0         # nothing sits below the lowest positive
    assert opts[1] == {"misses": 1, "lo": 0.01, "deep_read_band": 5}     # 0.02, 0.02, 0.03, 0.04, 0.05
    assert opts[2]["lo"] == 0.03 and opts[2]["deep_read_band"] == 2     # 0.04, 0.05
