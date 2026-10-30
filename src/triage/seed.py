"""Load the company lists into `companies`.

    python -m triage.seed               # AU shortlist + Gulf final_50
    python -m triage.seed --remote-first  # + data/remote_first.csv: remote-first employers (then run probe)
    python -m triage.seed --backfill    # + Greenhouse/Lever/Ashby companies from career-ops (volume only)

Run order: seed -> probe -> seed --backfill. The backfill skips a (board, slug) that an AU/Gulf
company already owns, so your own lists always come first.
"""
import argparse
import asyncio
import csv
import re
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from triage.ats import parse_ats_urls
from triage.db import SessionLocal, engine
from triage.models import Company

AU_CSV = Path("/mnt/data/au-ai-startups-research/data/shortlist.csv")
GULF_CSV = Path("/mnt/data/gulf-ai-startups-research/data/final_50.csv")
PORTALS = Path("/mnt/data/career-ops/portals.yml")
REMOTE_CSV = Path(__file__).resolve().parents[2] / "data" / "remote_first.csv"

_BACKFILL_BOARDS = {"greenhouse", "lever", "ashby"}
_ATS_HOST = re.compile(r"greenhouse\.io|lever\.co|ashbyhq\.com", re.I)
_DOMAIN = re.compile(r"^(https?://)?[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", re.I)


def clean_website(value: str) -> str | None:
    """The AU list has placeholders like 'unknown (search: ...)' where no domain was found."""
    value = value.strip()
    return value if _DOMAIN.match(value) else None


def read_companies(path: Path, region: str, name_col: str, site_col: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        return [
            {"name": r[name_col].strip(), "website": clean_website(r[site_col]), "region": region}
            for r in rows
            if r[name_col].strip()
        ]


def read_backfill(path: Path = PORTALS) -> list[dict]:
    """Enabled career-ops tracked_companies whose careers_url is a Greenhouse/Lever/Ashby board."""
    out = []
    for t in yaml.safe_load(path.read_text(encoding="utf-8"))["tracked_companies"]:
        if t.get("enabled") is False or not _ATS_HOST.search(t.get("careers_url", "")):
            continue
        parsed = [p for p in parse_ats_urls(t["careers_url"]) if p[0] in _BACKFILL_BOARDS]
        if parsed:
            board, slug = parsed[0]
            out.append({"name": t["name"].strip(), "website": None, "region": "BACKFILL", "board": board, "slug": slug})
    return out


async def upsert_companies(rows: list[dict]) -> int:
    """Insert new companies by name. Existing rows only get their website refreshed, never board/slug."""
    async with SessionLocal() as s, s.begin():
        before = len((await s.execute(select(Company.id))).all())
        for r in rows:
            stmt = insert(Company).values(**r)
            await s.execute(stmt.on_conflict_do_update(
                index_elements=[Company.name],
                set_={"website": stmt.excluded.website},
            ))
        return len((await s.execute(select(Company.id))).all()) - before


async def seed_backfill(rows: list[dict]) -> tuple[int, int]:
    """Insert backfill companies as verified boards, skipping names or boards already owned. Returns (added, skipped)."""
    async with SessionLocal() as s, s.begin():
        names = {n.lower() for n in (await s.scalars(select(Company.name))).all()}
        boards = {(b, sl) for b, sl in (await s.execute(select(Company.board, Company.slug))).all() if b}
        added = 0
        for r in rows:
            if r["name"].lower() in names or (r["board"], r["slug"]) in boards:
                continue
            s.add(Company(**r, verified=True))
            names.add(r["name"].lower())
            boards.add((r["board"], r["slug"]))
            added += 1
        return added, len(rows) - added


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-first", action="store_true", help="also load data/remote_first.csv (region REMOTE)")
    ap.add_argument("--backfill", action="store_true", help="also load career-ops Greenhouse/Lever/Ashby companies")
    args = ap.parse_args()

    rows = read_companies(AU_CSV, "AU", "company", "website") + read_companies(GULF_CSV, "GULF", "Company", "Website")
    new = await upsert_companies(rows)
    print(f"AU+GULF: {len(rows)} read, {new} new")
    if args.remote_first:
        rows = read_companies(REMOTE_CSV, "REMOTE", "company", "website")
        print(f"REMOTE: {len(rows)} read, {await upsert_companies(rows)} new")
    if args.backfill:
        added, skipped = await seed_backfill(read_backfill())
        print(f"BACKFILL: {added} added, {skipped} skipped (name or board already present)")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
