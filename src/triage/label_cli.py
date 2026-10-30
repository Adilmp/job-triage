"""Terminal labeler for the gold set.

    python -m triage.label_cli [--limit 150] [--region AU|GULF|BACKFILL] [--seed 1]
    python -m triage.label_cli --export        # write evals/gold.csv

Keys: f = fit, n = not fit, t = stretch (senior on paper, but worth applying), s = skip,
u = undo the last label (and show that post again), q = quit.
After f, n or t you may type a one-line note (enter to skip). Words starting with # in the note become tags
(for example "#research"), stored next to the label.

    python -m triage.label_cli --review        # second look at 'fit' labels that may break docs/FIT.md
    python -m triage.label_cli --include-prefiltered   # also label posts the prefilter would auto-drop

It NEVER shows Jev's or the LLM's output, or anything else a model produced, so the scores can't bias your
labels. Write what "fit" means in docs/FIT.md first: the tool won't start without it.
"""
import argparse
import asyncio
import csv
import random
import re
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from triage.db import SessionLocal, engine
from triage.models import Company, Label, Post
from triage.services.prefilter import REMOTE_HINT, prefilter, targets

ROOT = Path(__file__).resolve().parents[2]
FIT_FILE = ROOT / "docs" / "FIT.md"
GOLD_CSV = ROOT / "evals" / "gold.csv"
SHOW_CHARS = 1500
KEYS = {"f": "fit", "n": "not_fit", "t": "stretch"}
PRIORITY = {"AU": 0, "GULF": 0, "REMOTE": 1, "BACKFILL": 2}   # your own lists first, then remote-first employers; the backfill is volume


def parse_tags(note: str | None) -> list[str]:
    """#words in a note, lowercased, in order, no repeats. A # inside a word (C#) isn't a tag."""
    return list(dict.fromkeys(t.lower() for t in re.findall(r"(?<!\w)#(\w+)", note or "")))


_REMOTE_WORD = re.compile(r"remote|anywhere|distributed|home[- ]based|worldwide|global|work from home", re.I)
_REACHABLE = re.compile(r"asia|apac|south asia|global|worldwide|anywhere|pakistan|emea|any location", re.I)


def relevance_tier(location: str | None, remote: str, description: str) -> int:
    """How likely a post is to be one you could take, from plain facts about it (never a model's score).
    0 in Pakistan | 1 remote, no place named | 2 says remote (APAC/global/Pakistan if region-locked) |
    3 nothing says remote (a bare city) | 4 remote, but only inside a region that leaves you out."""
    loc = (location or "").lower()
    if any(t in loc for t in targets()):
        return 0
    if remote == "eligible":
        return 1
    if remote == "region_locked":
        return 2 if _REACHABLE.search(loc) else 4
    return 2 if _REMOTE_WORD.search(loc) or REMOTE_HINT.search(description or "") else 3


def _dropped(title, location, remote, description="") -> bool:
    return prefilter({"title": title, "location": location, "remote_eligibility": remote,
                      "description_text": description}).drop


async def unlabeled_ids(session, region: str | None = None, seed: int = 1, include_prefiltered: bool = False) -> list[int]:
    """Open, non-duplicate posts with a description and no gold label. Posts the prefilter would auto-drop are
    skipped (nobody needs to judge those), unless include_prefiltered. Your own AU/Gulf lists first; within a
    list, the posts most likely to be takeable first (relevance_tier), so you don't wade through San Francisco
    offices. Shuffled with a fixed seed inside each group so a session resumes in the same order."""
    q = (select(Post.id, Company.region, Post.title, Post.location, Post.remote_eligibility, Post.description_text).join(Company, Company.id == Post.company_id)
         .where(Post.closed_at.is_(None), Post.duplicate_of.is_(None), Post.description_text != "",
                ~select(Label.post_id).where(Label.post_id == Post.id, Label.source == "gold").exists())
         .order_by(Post.id))
    if region:
        q = q.where(Company.region == region)
    rows = (await session.execute(q)).all()
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[int]] = {}
    for pid, reg, title, location, remote, desc in rows:
        if include_prefiltered or not _dropped(title, location, remote, desc):
            groups.setdefault((PRIORITY[reg], relevance_tier(location, remote, desc)), []).append(pid)
    out = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        out += groups[key]
    return out


async def card(session, post_id: int, progress: str, extra: str = "") -> str:
    """What the labeler sees: title, company, location, link and the start of the description. Nothing a model made."""
    post, company = (await session.execute(
        select(Post, Company.name).join(Company, Company.id == Post.company_id).where(Post.id == post_id))).one()
    body = post.description_text[:SHOW_CHARS] + ("…" if len(post.description_text) > SHOW_CHARS else "")
    return (f"\n{'=' * 78}\n[{progress}]  {post.title}\n{company}  |  {post.location or 'no location'}\n{post.url}\n"
            f"{'-' * 78}\n{body}\n{'-' * 78}\n{extra}"
            f"Fit?  [f] fit   [n] not fit   [t] stretch   [s] skip   [u] undo   [q] quit")


async def save_label(session, post_id: int, label: str, notes: str | None) -> None:
    stmt = insert(Label).values(post_id=post_id, source="gold", label=label, notes=notes or None, tags=parse_tags(notes))
    await session.execute(stmt.on_conflict_do_update(
        index_elements=[Label.post_id, Label.source],
        set_={"label": stmt.excluded.label, "notes": stmt.excluded.notes, "tags": stmt.excluded.tags,
              "labeled_at": func.now()}))
    await session.commit()


async def undo_label(session, post_id: int) -> None:
    row = await session.get(Label, (post_id, "gold"))
    if row:
        await session.delete(row)
        await session.commit()


async def gold_count(session) -> int:
    return await session.scalar(select(func.count()).select_from(Label).where(Label.source == "gold"))


def _require_fit_definition() -> str:
    text = FIT_FILE.read_text().strip() if FIT_FILE.exists() else ""
    if not text or "TODO" in text:
        sys.exit(f"Write one sentence in {FIT_FILE} saying what a FIT is (role type, seniority, location/remote "
                 f"from Pakistan) before labeling, so the labels don't drift as you go.")
    return text


async def run_session(session, *, read_key, read_note, write, limit: int | None = None, region: str | None = None,
                      seed: int = 1, include_prefiltered: bool = False) -> None:
    write(f"FIT = {_require_fit_definition()}")
    queue = await unlabeled_ids(session, region, seed, include_prefiltered)
    target = limit or 150
    done: list[int] = []                      # labeled this session, for undo
    write(f"{await gold_count(session)} labeled so far, target {target}")
    while queue and await gold_count(session) < target:
        pid = queue.pop(0)
        write(await card(session, pid, f"{await gold_count(session) + 1}/{target}"))
        while True:
            key = read_key().lower()
            if key == "q":
                write(f"{await gold_count(session)} labeled")
                return
            if key == "s":
                break
            if key in KEYS:
                await save_label(session, pid, KEYS[key], read_note().strip())
                done.append(pid)
                break
            if key == "u":
                if not done:
                    write("nothing to undo")
                    continue
                last = done.pop()
                await undo_label(session, last)
                queue[:0] = [last, pid]      # show the undone post again, then the one on screen
                break
    write(f"{await gold_count(session)} labeled")


async def review_ids(session) -> list[int]:
    """Labels worth a second look, oldest first:
    - 'fit' or 'stretch' that carries a note (you hesitated), is research-titled, is region-locked remote, or that the prefilter drops;
    - 'not_fit' on a senior/lead engineering title that now gets past the prefilter (it may be a 'stretch')."""
    rows = (await session.execute(
        select(Label.post_id, Label.label, Label.notes, Post.title, Post.location, Post.remote_eligibility, Post.description_text)
        .join(Post, Post.id == Label.post_id).where(Label.source == "gold", Label.label.in_(["fit", "stretch", "not_fit"]))
        .order_by(Label.post_id))).all()
    out = []
    for pid, label, notes, title, loc, remote, desc in rows:
        dropped = _dropped(title, loc, remote, desc)
        if label in ("fit", "stretch") and ((notes or "").strip() or re.search(r"research", title, re.I)
                               or remote == "region_locked" or dropped):
            out.append(pid)
        elif label == "not_fit" and not dropped and re.search(r"\b(senior|sr|lead|iii)\b", title, re.I) \
                and re.search(r"engineer|developer|scientist|ml|ai\b|data", title, re.I):
            out.append(pid)
    return out


async def run_review(session, *, read_key, read_note, write) -> None:
    """Second look at suspicious labels. f / n / t re-label (with a fresh note), k or s keeps, q quits."""
    write(f"FIT = {_require_fit_definition()}")
    ids = await review_ids(session)
    for i, pid in enumerate(ids, 1):
        row = await session.get(Label, (pid, "gold"))
        extra = f"Current label: {row.label}   note: {row.notes or '-'}\n"
        write(await card(session, pid, f"review {i}/{len(ids)}", extra))
        while True:
            key = read_key().lower()
            if key == "q":
                return
            if key in ("k", "s"):
                break
            if key in KEYS:
                await save_label(session, pid, KEYS[key], read_note().strip())
                break
    write(f"reviewed {len(ids)} labels")


async def export_gold(session, path: Path = GOLD_CSV) -> int:
    rows = (await session.execute(
        select(Label.post_id, Label.label, Label.notes, Label.tags, Post.provider, Post.external_id)
        .join(Post, Post.id == Label.post_id).where(Label.source == "gold").order_by(Label.post_id))).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id", "label", "notes", "tags", "provider", "external_id"])
        w.writerows((pid, label, notes or "", ",".join(tags or []), prov, ext) for pid, label, notes, tags, prov, ext in rows)
    return len(rows)


def _read_key() -> str:
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q").strip()[:1] or "s"
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="stop once this many gold labels exist")
    ap.add_argument("--region", choices=["AU", "GULF", "REMOTE", "BACKFILL"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--export", action="store_true", help="write evals/gold.csv and exit")
    ap.add_argument("--review", action="store_true", help="second look at 'fit' labels that may break docs/FIT.md")
    ap.add_argument("--include-prefiltered", action="store_true", help="also show posts the prefilter would auto-drop")
    args = ap.parse_args()
    async with SessionLocal() as s:
        if args.export:
            print(f"wrote {await export_gold(s)} labels to {GOLD_CSV}")
        elif args.review:
            await run_review(s, read_key=_read_key, read_note=lambda: input("note (enter to skip): "), write=print)
        else:
            await run_session(s, read_key=_read_key, read_note=lambda: input("note (enter to skip): "), write=print,
                              limit=args.limit, region=args.region, seed=args.seed,
                              include_prefiltered=args.include_prefiltered)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
