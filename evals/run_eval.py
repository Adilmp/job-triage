"""Measure the triage pipeline against your gold labels.

    uv run python evals/run_eval.py split                  # once: freeze a stratified tuning / held-out split
    uv run python evals/run_eval.py tune                   # Jev on the tuning split only -> suggested HI and LO
    uv run python evals/run_eval.py run --split tuning     # the whole pipeline on the tuning split
    uv run python evals/run_eval.py run --split heldout --save-baseline   # ONCE, after HI/LO are chosen

How a run works: every gold post goes through the real graph (prefilter, Jev, route, deep read). When the graph
pauses for a human, the gold label answers for you: approve if you labeled it fit (or stretch), reject otherwise.
Nothing is written to the database: the graph's two writes (set_status, record_run) are swapped for no-ops.

Jev and deep-read answers are cached in evals/cache/ by (wording, model, post text), so a re-run costs nothing
and a wording change re-asks only what changed. Latency and cost are reported from when each answer was first
computed, not from the cache hit.

Honesty rules this file enforces:
  - HI and LO are chosen on the tuning split (`tune`). The held-out split is for measuring, once.
    Every held-out run is logged in evals/heldout_log.csv and a repeat prints a warning.
  - Every rate comes with a 95% Wilson interval. With ~100 held-out posts a bare percentage overstates what is known.
"""
import argparse
import asyncio
import contextvars
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from triage import graph as graph_mod
from triage import repo
from triage.config import settings
from triage.criteria import criteria_version
from triage.db import engine
from triage.services import deep_read as deep_read_mod
from triage.services import jev
from triage.services.deep_read import DeepRead
from triage.services.prefilter import prefilter

EVALS = Path(__file__).resolve().parent
GOLD = EVALS / "gold.csv"
SPLIT = EVALS / "split.csv"
CACHE = EVALS / "cache" / "answers.json"
RUNS = EVALS / "runs"
BASELINE = EVALS / "baseline.json"
HELDOUT_LOG = EVALS / "heldout_log.csv"

SEED = 7
HELDOUT_FRAC = 0.4          # 250 labels -> 150 tuning, 100 held out
CONCURRENCY = 4             # Jev calls in flight; the deep read is one at a time anyway


# =====================================================================================================
# Metrics. Plain functions on plain numbers, so each one is easy to test and to explain.
# =====================================================================================================

def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """k successes out of n, with a 95% Wilson score interval. Better than p +/- 1.96*sqrt(p(1-p)/n) at small n
    and near 0 or 1, where the plain formula gives impossible intervals like [-0.05, 0.12]."""
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"k": k, "n": n, "p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def auc(scores: list[float], positive: list[bool]) -> float | None:
    """Probability that a random positive scores above a random negative (ties count half). This is the
    Mann-Whitney U statistic divided by n_pos * n_neg. None if either class is missing."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for value in np.unique(s):              # average the ranks of tied scores
        tied = s == value
        ranks[tied] = ranks[tied].mean()
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def pick_thresholds(scores: list[float], positive: list[bool], min_precision: float = 0.8, min_support: int = 3) -> dict:
    """Suggest LO and HI from tuning-split Jev scores.
    LO: as high as possible while no positive scores at or below it (a false reject is the costly error).
    HI: the lowest score above which at least `min_precision` of posts are positive, with `min_support` posts.
    Both are only suggestions: look at the table before setting them."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    candidates = np.unique(np.round(s, 3))
    pos_scores = s[y]
    below_all_pos = candidates[candidates < pos_scores.min()] if len(pos_scores) else candidates
    lo = float(below_all_pos.max()) if len(below_all_pos) else 0.0
    hi = None
    for t in candidates:
        above = s >= t
        if above.sum() >= min_support and y[above].mean() >= min_precision and t > lo:
            hi = float(t)
            break
    return {"lo": lo, "hi": hi if hi is not None else 1.0}


def lo_options(scores: list[float], positive: list[bool], hi: float, max_misses: int = 3) -> list[dict]:
    """For 0..max_misses allowed false rejects at the Jev stage: the highest LO that allows them, and how many posts
    would then still need the (slow) deep read. Shows what each saved deep read costs in missed fits."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    pos = np.sort(s[y])
    out = []
    for k in range(max_misses + 1):
        if k >= len(pos):
            break
        nxt = pos[k]                                    # the (k+1)-th lowest positive must stay above LO
        below = np.unique(np.round(s[s < nxt], 3))
        lo = float(below.max()) if len(below) else 0.0
        option = {"misses": int((pos <= lo).sum()), "lo": lo, "deep_read_band": int(((s > lo) & (s < hi)).sum())}
        if option not in out:                           # tied positive scores give the same option twice
            out.append(option)
    return out


def threshold_table(scores: list[float], positive: list[bool], steps=None) -> list[dict]:
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    rows = []
    for t in steps if steps is not None else np.round(np.arange(0.05, 1.0, 0.05), 2):
        above, at_or_below = s >= t, s <= t
        rows.append({"t": float(t), "n_above": int(above.sum()),
                     "precision_above": float(y[above].mean()) if above.any() else None,
                     "recall_above": float(y[above].sum() / y.sum()) if y.any() else None,
                     "positives_at_or_below": int(y[at_or_below].sum())})
    return rows


def report(records: list[dict], hi: float, lo: float) -> dict:
    """Every number in the eval, computed from one list of per-post records."""
    n = len(records)
    pos = [r for r in records if r["positive"]]
    scored = [r for r in records if r["jev_p"] is not None]
    scored_pos = [r for r in scored if r["positive"]]

    jev_alone = {
        "auc": auc([r["jev_p"] for r in scored], [r["positive"] for r in scored]),
        "at_hi": {"precision": wilson(sum(r["positive"] for r in scored if r["jev_p"] >= hi), sum(r["jev_p"] >= hi for r in scored)),
                  "recall": wilson(sum(r["jev_p"] >= hi for r in scored_pos), len(scored_pos))},
        "at_lo": {"false_reject": wilson(sum(r["jev_p"] <= lo for r in scored_pos), len(scored_pos))},
    }
    routes = ("prefilter_drop", "auto_reject", "deep_read", "shortlist")
    router = {route: wilson(sum(r["route"] == route for r in records), n) for route in routes}

    missed = [r for r in pos if not r["proposed"]]
    false_reject = {**wilson(len(missed), len(pos)),
                    "by_stage": {s: sum(_stage(r) == s for r in missed) for s in ("prefilter", "jev_low", "deep_read")}}

    band = [r for r in records if r["route"] == "deep_read"]
    mid = (hi + lo) / 2
    deep = {"n": len(band),
            "pipeline_accuracy": wilson(sum(r["proposed"] == r["positive"] for r in band), len(band)),
            "jev_alone_accuracy": wilson(sum((r["jev_p"] >= mid) == r["positive"] for r in band), len(band)),
            "jev_alone_cut": mid}

    proposed = [r for r in records if r["proposed"]]
    ms = [r["model_ms"] for r in records]
    end_to_end = {
        "precision": wilson(sum(r["positive"] for r in proposed), len(proposed)),
        "recall": wilson(sum(r["proposed"] for r in pos), len(pos)),
        "review_per_100": 100 * len(proposed) / n if n else None,
        "latency_ms": {"p50": float(np.percentile(ms, 50)) if ms else None, "p95": float(np.percentile(ms, 95)) if ms else None},
        "jev_cost_usd": sum(r["cost_usd"] for r in records),
        "jev_cost_per_100_usd": 100 * sum(r["cost_usd"] for r in records) / n if n else None,
    }
    false_alarms = sorted((r for r in proposed if not r["positive"]), key=lambda r: -(r["jev_p"] or 0))
    return {"n": n, "positives": len(pos), "jev_alone": jev_alone, "router": router, "false_reject": false_reject,
            "deep_read": deep, "end_to_end": end_to_end,
            "failures": {"false_rejects": [_failure(r) for r in sorted(missed, key=lambda r: -(r["jev_p"] or 0))][:20],
                         "false_alarms": [_failure(r) for r in false_alarms][:20]}}


def _stage(r: dict) -> str:
    return {"prefilter_drop": "prefilter", "auto_reject": "jev_low"}.get(r["route"], "deep_read")


def _failure(r: dict) -> dict:
    dr = r.get("deep_read") or {}
    return {"post_id": r["post_id"], "title": r["title"], "company": r["company"], "label": r["label"],
            "tags": r["tags"], "route": r["route"], "stage": _stage(r), "jev_p": r["jev_p"], "why": r.get("why"),
            "prefilter_reason": r.get("reason"), "deep_read_reasons": dr.get("reasons"), "red_flags": dr.get("red_flags"),
            "deep_read_confidence": dr.get("confidence")}


def gate(current: dict, baseline: dict | None) -> list[str]:
    """Regression gate: fail if the false-reject rate rose above the baseline's interval, or recall fell below it."""
    if baseline is None:
        return []
    problems = []
    fr, b_fr = current["false_reject"], baseline["false_reject"]
    if fr["p"] is not None and b_fr["hi"] is not None and fr["p"] > b_fr["hi"]:
        problems.append(f"false-reject rate {fr['p']:.1%} is above the baseline's upper bound {b_fr['hi']:.1%}")
    rc, b_rc = current["end_to_end"]["recall"], baseline["end_to_end"]["recall"]
    if rc["p"] is not None and b_rc["lo"] is not None and rc["p"] < b_rc["lo"]:
        problems.append(f"recall {rc['p']:.1%} is below the baseline's lower bound {b_rc['lo']:.1%}")
    return problems


# =====================================================================================================
# Gold labels and the split
# =====================================================================================================

def load_gold(path: Path = GOLD, stretch_positive: bool = True) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{"post_id": int(r["post_id"]), "label": r["label"], "tags": [t for t in (r.get("tags") or "").split(",") if t],
             "positive": r["label"] == "fit" or (stretch_positive and r["label"] == "stretch")} for r in rows]


def make_split(gold: list[dict], seed: int = SEED, heldout_frac: float = HELDOUT_FRAC) -> dict[int, str]:
    """Stratified by label (fit / stretch / not_fit), so both halves have the same mix. Seeded, so it's reproducible."""
    rng = random.Random(seed)
    out = {}
    for label in sorted({g["label"] for g in gold}):
        ids = sorted(g["post_id"] for g in gold if g["label"] == label)
        rng.shuffle(ids)
        k = round(len(ids) * heldout_frac)
        out |= {pid: "heldout" for pid in ids[:k]} | {pid: "tuning" for pid in ids[k:]}
    return out


def write_split(split: dict[int, str], path: Path = SPLIT, force: bool = False) -> None:
    if path.exists() and not force:
        sys.exit(f"{path} already exists. The split is frozen on purpose; pass --force only if you mean to redo it.")
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id", "split"])
        w.writerows(sorted(split.items()))


def read_split(path: Path = SPLIT) -> dict[int, str]:
    if not path.exists():
        sys.exit(f"No {path}. Run `python evals/run_eval.py split` first.")
    with path.open(newline="") as fh:
        return {int(r["post_id"]): r["split"] for r in csv.DictReader(fh)}


# =====================================================================================================
# Running the real graph with cached model answers and no database writes
# =====================================================================================================

class AnswerCache:
    def __init__(self, path: Path = CACHE):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


_current: contextvars.ContextVar[dict] = contextvars.ContextVar("current_eval_record")


class Instrumented:
    """Swaps in cached Jev / deep-read calls and no-op database writes for the duration of a run, then puts the
    originals back. Each model answer adds its first-computed time and cost to the post being evaluated."""

    def __init__(self, cache: AnswerCache):
        self.cache, self.fresh_jev, self.fresh_deep, self.spent_usd = cache, 0, 0, 0.0

    def __enter__(self):
        self._saved = (jev.ask, graph_mod.deep_read, repo.set_status, repo.record_run)
        real_ask_with_usage, real_deep_read = jev.ask_with_usage, self._saved[1]

        async def ask(text, questions=jev.QUESTIONS, client=None):
            key = _key("jev", questions, settings.JEV_MODEL, text)
            if key not in self.cache.data:
                start = time.monotonic()
                probs, usage = await real_ask_with_usage(text, questions, client)
                cost = jev.cost_usd(usage)
                self.cache.data[key] = {"probs": probs, "ms": (time.monotonic() - start) * 1000, "cost": cost,
                                        "model": usage.get("model")}
                self.fresh_jev += 1
                self.spent_usd += cost
            hit = self.cache.data[key]
            rec = _current.get()
            rec["model_ms"] += hit["ms"]
            rec["cost_usd"] += hit["cost"]
            return hit["probs"]

        async def deep_read(post):
            key = _key("deep", deep_read_mod.SYSTEM_PROMPT, settings.LLM_MODEL, deep_read_mod._human(post))
            if key not in self.cache.data:
                start = time.monotonic()
                dump = (await real_deep_read(post)).model_dump()
                self.cache.data[key] = {"deep": dump, "ms": (time.monotonic() - start) * 1000}
                self.fresh_deep += 1
                self.cache.save()          # deep reads are slow: never lose one to a crash
            hit = self.cache.data[key]
            _current.get()["model_ms"] += hit["ms"]
            return DeepRead(**hit["deep"])

        async def no_write(*a, **kw):
            return 0

        jev.ask, graph_mod.deep_read, repo.set_status, repo.record_run = ask, deep_read, no_write, no_write
        return self

    def __exit__(self, *exc):
        jev.ask, graph_mod.deep_read, repo.set_status, repo.record_run = self._saved
        self.cache.save()


async def evaluate(gold: list[dict], hi: float, lo: float, cache: AnswerCache, progress=print) -> tuple[list[dict], dict]:
    """Run each gold post through the graph; return per-post records and the run's settings."""
    saved = (settings.HI, settings.LO)
    settings.HI, settings.LO = hi, lo                     # HI/LO are part of criteria_version, so set them first
    try:
        cv = criteria_version()
        g = await graph_mod.build_graph(InMemorySaver())
        sem = asyncio.Semaphore(CONCURRENCY)
        with Instrumented(cache) as inst:
            async def one(item: dict) -> dict:
                post = await repo.get_post_dict(item["post_id"])
                if post is None:
                    raise SystemExit(f"gold post {item['post_id']} is not in the database")
                rec = {**item, "title": post["title"], "company": post["company"], "model_ms": 0.0, "cost_usd": 0.0}
                _current.set(rec)
                cfg = {"configurable": {"thread_id": f"eval-{item['post_id']}-{uuid.uuid4().hex}"}}
                async with sem:
                    out = await g.ainvoke({"post_id": post["id"], "post": post, "criteria_version": cv}, cfg)
                    if out.get("__interrupt__"):         # the gold label answers for the human
                        out = await g.ainvoke(Command(resume="approve" if item["positive"] else "reject"), cfg)
                rec.update({"route": out["route"], "jev_p": out.get("jev_p"), "proposed": bool(out.get("proposed")),
                            "why": out.get("why"), "deep_read": out.get("deep_read"), "reason": out.get("reason"),
                            "final_status": out["status"]})
                progress(f"  {item['post_id']:>6} {rec['label']:8} -> {rec['route']:14} jev={rec['jev_p'] if rec['jev_p'] is not None else '-'}")
                return rec

            records = await asyncio.gather(*(one(item) for item in gold))
        meta = {"criteria_version": cv, "hi": hi, "lo": lo, "min_conf": settings.DEEP_READ_MIN_CONFIDENCE,
                "jev_model": settings.JEV_MODEL, "llm_model": settings.LLM_MODEL,
                "fresh_jev_calls": inst.fresh_jev, "fresh_deep_reads": inst.fresh_deep, "spent_usd_this_run": inst.spent_usd}
        return list(records), meta
    finally:
        settings.HI, settings.LO = saved


async def jev_scores(gold: list[dict], cache: AnswerCache) -> list[dict]:
    """Jev scores only (no graph, no deep read), for posts that pass the prefilter: what `tune` needs."""
    sem = asyncio.Semaphore(CONCURRENCY)
    with Instrumented(cache):
        async def one(item):
            post = await repo.get_post_dict(item["post_id"])
            rec = {**item, "title": post["title"], "model_ms": 0.0, "cost_usd": 0.0, "dropped": prefilter(post).drop}
            if rec["dropped"]:
                return rec
            _current.set(rec)
            async with sem:
                rec["jev_p"] = (await jev.ask(jev.post_text(post)))["fit"]
            return rec
        return list(await asyncio.gather(*(one(i) for i in gold)))


# =====================================================================================================
# CLI
# =====================================================================================================

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=EVALS).stdout.strip()
    except Exception:
        return "nogit"


def _fmt(w: dict) -> str:
    if w.get("p") is None:
        return "n/a (n=0)"
    return f"{w['p']:.1%}  [{w['lo']:.1%}, {w['hi']:.1%}]  ({w['k']}/{w['n']})"


def print_report(rep: dict, meta: dict) -> None:
    j, e, fr, d = rep["jev_alone"], rep["end_to_end"], rep["false_reject"], rep["deep_read"]
    print(f"\n{rep['n']} posts, {rep['positives']} positive | HI={meta['hi']} LO={meta['lo']} min_conf={meta['min_conf']} "
          f"| criteria {meta['criteria_version']}")
    print(f"\nJev alone        AUC {j['auc']:.3f}" if j["auc"] is not None else "\nJev alone        AUC n/a")
    print(f"  at HI           precision {_fmt(j['at_hi']['precision'])}   recall {_fmt(j['at_hi']['recall'])}")
    print(f"  at LO           false rejects {_fmt(j['at_lo']['false_reject'])}")
    print("\nRouter (share of all posts)")
    for route, w in rep["router"].items():
        print(f"  {route:16}{_fmt(w)}")
    print(f"\nFALSE-REJECT RATE {_fmt(fr)}   by stage: {fr['by_stage']}")
    print(f"\nDeep read on the uncertain band (n={d['n']})")
    print(f"  pipeline        {_fmt(d['pipeline_accuracy'])}")
    print(f"  Jev alone       {_fmt(d['jev_alone_accuracy'])}   (cut at {d['jev_alone_cut']:.2f})")
    print("\nEnd to end")
    print(f"  precision       {_fmt(e['precision'])}")
    print(f"  recall          {_fmt(e['recall'])}")
    print(f"  review load     {e['review_per_100']:.1f} posts per 100 reach you")
    print(f"  latency         p50 {e['latency_ms']['p50']:.0f} ms, p95 {e['latency_ms']['p95']:.0f} ms (model time)")
    print(f"  Jev cost        ${e['jev_cost_usd']:.5f} total, ${e['jev_cost_per_100_usd']:.5f} per 100 posts "
          f"(spent now: ${meta['spent_usd_this_run']:.5f}, {meta['fresh_jev_calls']} new Jev calls, {meta['fresh_deep_reads']} new deep reads)")
    if rep["failures"]["false_rejects"]:
        print("\nFalse rejects (gold positive, never reached you): add each new kind to docs/KNOWN_ISSUES.md")
        for f in rep["failures"]["false_rejects"]:
            why = f["prefilter_reason"] or (f["red_flags"] or f["deep_read_reasons"] or "")
            print(f"  [{f['stage']:9}] jev={f['jev_p'] if f['jev_p'] is not None else '-'}  {f['title'][:45]} @ {f['company']}  {why}")


def cmd_split(args) -> None:
    gold = load_gold()
    split = make_split(gold)
    write_split(split, force=args.force)
    for name in ("tuning", "heldout"):
        part = [g for g in gold if split[g["post_id"]] == name]
        print(f"{name:8} {len(part):4} posts, {sum(g['positive'] for g in part)} positive")


async def cmd_tune(args) -> None:
    split = read_split()
    gold = [g for g in load_gold(stretch_positive=not args.stretch_negative) if split.get(g["post_id"]) == "tuning"]
    cache = AnswerCache()
    recs = await jev_scores(gold, cache)
    dropped_pos = [r for r in recs if r["dropped"] and r["positive"]]
    scored = [r for r in recs if not r["dropped"]]
    s, y = [r["jev_p"] for r in scored], [r["positive"] for r in scored]
    print(f"tuning split: {len(recs)} posts, {len(scored)} pass the prefilter, AUC {auc(s, y):.3f}")
    if dropped_pos:
        print(f"  {len(dropped_pos)} positive(s) dropped by the prefilter before Jev: {[r['title'][:30] for r in dropped_pos]}")
    print(f"\n{'t':>5} {'n>=t':>6} {'prec>=t':>8} {'recall>=t':>10} {'pos<=t':>7}")
    for row in threshold_table(s, y):
        prec = f"{row['precision_above']:.2f}" if row["precision_above"] is not None else "-"
        rec = f"{row['recall_above']:.2f}" if row["recall_above"] is not None else "-"
        print(f"{row['t']:>5.2f} {row['n_above']:>6} {prec:>8} {rec:>10} {row['positives_at_or_below']:>7}")
    pick = pick_thresholds(s, y, min_precision=args.min_precision)
    print(f"\nHI={pick['hi']}: the lowest score where at least {args.min_precision:.0%} of posts above it are positive.")
    print("LO options (each saved deep read can cost a missed fit):")
    for o in lo_options(s, y, pick["hi"]):
        print(f"  LO={o['lo']:<6} misses {o['misses']}/{sum(y)} positives at the Jev stage, "
              f"{o['deep_read_band']}/{len(s)} posts go to the deep read (~{o['deep_read_band']} min of local LLM)")
    print("Pick HI and LO, set them in .env, then run the held-out split once.")
    await engine.dispose()


async def cmd_run(args) -> None:
    split = read_split()
    gold = [g for g in load_gold(stretch_positive=not args.stretch_negative) if split.get(g["post_id"]) == args.split]
    hi = args.hi if args.hi is not None else settings.HI
    lo = args.lo if args.lo is not None else settings.LO
    if args.split == "heldout":
        _log_heldout(hi, lo)
    print(f"running {len(gold)} {args.split} posts through the graph (HI={hi}, LO={lo})")
    records, meta = await evaluate(gold, hi, lo, AnswerCache(), progress=print if args.verbose else (lambda *_: None))
    rep = report(records, hi, lo)
    meta |= {"split": args.split, "stretch_positive": not args.stretch_negative, "git": _git_sha(),
             "at": datetime.now(timezone.utc).isoformat()}
    print_report(rep, meta)

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    problems = gate(rep, baseline)
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{meta['git']}.json"
    out.write_text(json.dumps({"meta": meta, "report": rep, "gate": problems, "records": records}, indent=1, default=str))
    print(f"\nwrote {out}")
    if args.save_baseline:
        BASELINE.write_text(json.dumps({"meta": meta, "false_reject": rep["false_reject"], "end_to_end": rep["end_to_end"]}, indent=1))
        print(f"saved as the baseline: {BASELINE}")
    elif baseline is None:
        print("no baseline yet: re-run with --save-baseline to set one")
    if problems:
        print("\nGATE FAILED:\n  " + "\n  ".join(problems))
        await engine.dispose()
        sys.exit(1)
    print("gate: pass" if baseline else "")
    await engine.dispose()


def _log_heldout(hi: float, lo: float) -> None:
    prior = list(csv.DictReader(HELDOUT_LOG.open())) if HELDOUT_LOG.exists() else []
    if prior:
        print(f"WARNING: the held-out split has been run {len(prior)} time(s) before. If you changed HI, LO or the wording "
              f"because of those numbers, this run is no longer an unbiased measurement. Say so in the README.")
    new = not HELDOUT_LOG.exists()
    with HELDOUT_LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["at", "criteria_version", "hi", "lo"])
        settings_hi, settings_lo = settings.HI, settings.LO
        settings.HI, settings.LO = hi, lo
        w.writerow([datetime.now(timezone.utc).isoformat(), criteria_version(), hi, lo])
        settings.HI, settings.LO = settings_hi, settings_lo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="freeze the tuning / held-out split")
    sp.add_argument("--force", action="store_true")
    for name in ("tune", "run"):
        p = sub.add_parser(name)
        p.add_argument("--stretch-negative", action="store_true", help="count 'stretch' labels as not fit")
        if name == "tune":
            p.add_argument("--min-precision", type=float, default=0.8)
        else:
            p.add_argument("--split", choices=["tuning", "heldout"], required=True)
            p.add_argument("--hi", type=float)
            p.add_argument("--lo", type=float)
            p.add_argument("--save-baseline", action="store_true")
            p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.cmd == "split":
        cmd_split(args)
    else:
        asyncio.run(cmd_tune(args) if args.cmd == "tune" else cmd_run(args))


if __name__ == "__main__":
    main()
