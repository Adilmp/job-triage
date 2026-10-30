"""Find each company's ATS board and mark it verified or not.

    python -m triage.probe               # companies with no board yet
    python -m triage.probe --backfill    # liveness-check BACKFILL rows (their slug came from a URL)
    python -m triage.probe --report      # print the per-region table from the DB, no network

Only 'high' trust matches (slug is the company's own domain stem, or the board was linked from
the company's own careers page) get verified=true. Guesses are stored with verified=false so you
can review them; the fetch cron skips them. Ported from gulf-ai-startups-research/probe_ats.py.
"""
import argparse
import asyncio
import re
import time
from dataclasses import dataclass

import httpx
from sqlalchemy import select

from triage.ats import parse_ats_urls, slugs, trust_of
from triage.db import SessionLocal, engine
from triage.models import Company

UA = {"User-Agent": "Mozilla/5.0 (compatible; job-triage/0.1)"}
PROVIDERS = ("ashby", "lever", "greenhouse", "recruitee", "workable", "teamtailor")


@dataclass
class Candidate:
    board: str
    slug: str
    trust: str
    n_jobs: int
    via: str = "slug"


_host_slots: dict[str, asyncio.Semaphore] = {}
_blocked_until: dict[str, float] = {}   # site -> monotonic time a long Retry-After lifts


# Requests per run a site tolerates from us. apply.workable.com banned this IP for 24h twice after probe runs of
# a few hundred requests, so past the budget it is treated like a rate-limited site (answers become "unknown").
_BUDGET = {"workable.com": 40}
_spent: dict[str, int] = {}


def _site(url: str) -> str:
    host = httpx.URL(url).host
    return ".".join(host.split(".")[-2:]) if host else ""


def rate_limited_sites() -> list[str]:
    now = time.monotonic()
    return sorted(site for site, until in _blocked_until.items() if until > now)


async def _get(client: httpx.AsyncClient, url: str, tries: int = 4) -> httpx.Response | None:
    """200 response or None. 429/5xx and transport errors are retried with backoff; a 404 means no such board.
    A 429 whose Retry-After is long (apply.workable.com asks for ~24h after a burst) trips a breaker: the site
    is not contacted again this run, so we don't extend the ban, and report() says the answer is unknown."""
    site = _site(url)
    for attempt in range(tries):
        if _blocked_until.get(site, 0) > time.monotonic():
            return None
        if site in _BUDGET:
            _spent[site] = _spent.get(site, 0) + 1
            if _spent[site] > _BUDGET[site]:
                _blocked_until[site] = time.monotonic() + 3600
                return None
        try:
            async with _host_slots.setdefault(site, asyncio.Semaphore(3)):   # at most 3 in flight per site
                r = await client.get(url)
        except httpx.InvalidURL:
            return None
        except httpx.HTTPError:
            r = None
        if r is not None and r.status_code == 429:
            wait = int(r.headers.get("retry-after", 0) or 0)
            if wait > 120:
                if _blocked_until.get(site, 0) <= time.monotonic():
                    print(f"  !! {site} rate-limited us for {wait // 3600}h{wait % 3600 // 60:02d}m; skipping it for the rest of this run")
                _blocked_until[site] = time.monotonic() + wait
                return None
        if r is not None and r.status_code < 500 and r.status_code != 429:
            return r if r.status_code == 200 else None
        await asyncio.sleep(1.5 * 2**attempt)
    return None


async def _json(client: httpx.AsyncClient, url: str):
    r = await _get(client, url)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


async def board_info(client: httpx.AsyncClient, provider: str, slug: str) -> tuple[int, str | None] | None:
    """(open jobs, company name the ATS reports or None), or None if no such board. An empty board is (0, ...)."""
    if provider == "ashby":
        d = await _json(client, f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
        return (len(d["jobs"]), None) if isinstance(d, dict) and d.get("jobs") is not None else None
    if provider == "lever":
        d = await _json(client, f"https://api.lever.co/v0/postings/{slug}?mode=json")
        return (len(d), None) if isinstance(d, list) else None
    if provider == "greenhouse":
        d = await _json(client, f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")
        if not (isinstance(d, dict) and "jobs" in d):
            return None
        meta = await _json(client, f"https://boards-api.greenhouse.io/v1/boards/{slug}")
        return len(d["jobs"]), meta.get("name") if isinstance(meta, dict) else None
    if provider == "recruitee":
        d = await _json(client, f"https://{slug}.recruitee.com/api/offers/")
        if not (isinstance(d, dict) and "offers" in d):
            return None
        return len(d["offers"]), next((o["company_name"] for o in d["offers"] if o.get("company_name")), None)
    if provider == "workable":
        d = await _json(client, f"https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true")
        return (len(d["jobs"]), d.get("name")) if isinstance(d, dict) and "jobs" in d else None
    if provider == "teamtailor":
        # A slug with a dot is a custom career-site host (careers.acme.co), which serves the same feed.
        host = slug if "." in slug else f"{slug}.teamtailor.com"
        r = await _get(client, f"https://{host}/jobs.rss")
        if r is not None and "<channel" in r.text:
            title = re.search(r"<title>(?:<!\[CDATA\[)?\s*([^<\]]+?)\s*(?:\]\]>)?</title>", r.text)   # first <title> is the feed's
            return r.text.count("<item"), title.group(1) if title else None
        # No RSS: accept the HTML page only if we stayed on this slug's own teamtailor host.
        r = await _get(client, f"https://{host}/jobs")
        if r is not None and r.url.host == host and "teamtailor" in r.text.lower():
            return len(re.findall(r'href="[^"]*/jobs/\d+', r.text)), None
    return None


async def _probe_slug(client, slug: str, name: str, website: str | None) -> list[Candidate]:
    infos = await asyncio.gather(*(board_info(client, p, slug) for p in PROVIDERS))
    return [Candidate(p, slug, trust_of(slug, website, name, info[1]), info[0])
            for p, info in zip(PROVIDERS, infos) if info is not None]


async def _own_site_boards(client, website: str | None, careers_url: str | None) -> list[Candidate]:
    """Boards linked from the company's own site. Catches custom subdomains (careers.acme.co -> Teamtailor)
    that the slug probes never see. Trusted because the company itself links to it."""
    if not website:
        return []
    dom = re.sub(r"^https?://(www\.)?", "", website.strip().lower()).split("/")[0]
    urls = [u for u in (careers_url, f"https://careers.{dom}", f"https://jobs.{dom}", f"https://{dom}/careers") if u]
    for url in urls:
        r = await _get(client, url)
        if r is None:
            continue
        for board, slug in parse_ats_urls(r.text):
            info = await board_info(client, board, slug)
            if info is not None:
                return [Candidate(board, slug, "high", info[0], via=url)]
        if "teamtailor-cdn.com" in r.text and r.url.host.endswith(dom):
            # Teamtailor on the company's own domain: no teamtailor.com slug appears in the page.
            info = await board_info(client, "teamtailor", r.url.host)
            if info is not None:
                return [Candidate("teamtailor", r.url.host, "high", info[0], via=url)]
    return []


async def probe_company(client, name: str, website: str | None, careers_url: str | None = None) -> list[Candidate]:
    # Slugs from a bare ATS careers_url are the researcher's guess; they still have to pass trust_of().
    hints = [s for _, s in parse_ats_urls(careers_url or "")]
    for slug in dict.fromkeys(hints + slugs(name, website)):
        found = await _probe_slug(client, slug, name, website)
        if found:
            return found
    return await _own_site_boards(client, website, careers_url)


def pick(found: list[Candidate]) -> tuple[Candidate | None, str | None]:
    """Prefer a high-trust board; note the others so a human can review them."""
    if not found:
        return None, "no board found"
    best = sorted(found, key=lambda c: c.trust != "high")[0]
    others = [f"{c.board}:{c.slug}" for c in found if c is not best]
    note = None if best.trust == "high" and not others else "; ".join(
        ([f"unverified slug guess ({best.board}:{best.slug}), check it is really this company"] if best.trust != "high" else [])
        + ([f"also found {', '.join(others)}"] if others else []))
    return best, note


def _careers_urls() -> dict[str, str]:
    import csv
    from triage.seed import GULF_CSV
    with GULF_CSV.open(newline="", encoding="utf-8") as fh:
        return {r["Company"].strip(): r["Careers_URL"].strip() for r in csv.DictReader(fh) if r["Careers_URL"].strip()}


async def save_result(session, company_id: int, cand: Candidate | None, note: str | None, backfill: bool = False) -> None:
    """Store a probe result. A board another company already owns is not claimed twice: two rows on one board
    would fight over the same (provider, external_id) posts."""
    row = await session.get(Company, company_id)
    if cand is not None:
        owner = await session.scalar(select(Company.name).where(
            Company.board == cand.board, Company.slug == cand.slug, Company.id != company_id))
        if owner:
            row.last_error = f"board {cand.board}:{cand.slug} already belongs to {owner}"
            return
        row.board, row.slug, row.verified = cand.board, cand.slug, cand.trust == "high"
    elif backfill:
        row.verified = False
    row.last_error = note


async def run_probe(workers: int, backfill: bool, region: str | None = None) -> None:
    async with SessionLocal() as s:
        q = select(Company).order_by(Company.id)
        q = q.where(Company.region == "BACKFILL") if backfill else q.where(Company.board.is_(None), Company.region != "BACKFILL")
        if region:
            q = q.where(Company.region == region)
        companies = list((await s.scalars(q)).all())
    careers = _careers_urls()
    sem = asyncio.Semaphore(workers)
    results: dict[int, tuple[Candidate | None, str | None]] = {}

    async with httpx.AsyncClient(headers=UA, timeout=12, follow_redirects=True) as client:
        async def one(c: Company) -> None:
            async with sem:
                if backfill:
                    info = await board_info(client, c.board, c.slug)
                    cand = Candidate(c.board, c.slug, "high", info[0]) if info is not None else None
                    results[c.id] = (cand, None if cand else f"backfill board {c.board}:{c.slug} not found")
                else:
                    results[c.id] = pick(await probe_company(client, c.name, c.website, careers.get(c.name)))
            cand, note = results[c.id]
            print(f"  {c.region:8} {c.name[:30]:30} {f'{cand.board}:{cand.slug} [{cand.trust}] jobs={cand.n_jobs}' if cand else '-':50}")

        await asyncio.gather(*(one(c) for c in companies))

    async with SessionLocal() as s, s.begin():
        for c in companies:
            await save_result(s, c.id, *results[c.id], backfill=backfill)
    await report()


async def report() -> None:
    async with SessionLocal() as s:
        rows = (await s.scalars(select(Company))).all()
    print(f"\n{'region':9}{'companies':>10}{'verified':>10}{'guess':>7}{'none':>6}   verified by board")
    for region in ("AU", "GULF", "REMOTE", "BACKFILL"):
        rs = [r for r in rows if r.region == region]
        ver = [r for r in rs if r.verified]
        by = {}
        for r in ver:
            by[r.board] = by.get(r.board, 0) + 1
        print(f"{region:9}{len(rs):>10}{len(ver):>10}{sum(1 for r in rs if r.board and not r.verified):>7}"
              f"{sum(1 for r in rs if not r.board):>6}   {by}")
    if rate_limited_sites():
        print(f"\nNOTE: {', '.join(rate_limited_sites())} rate-limited this run; boards on it are unknown, not absent. Re-run later.")
    guesses = [r for r in rows if r.board and not r.verified and r.region != "BACKFILL"]
    if guesses:
        print("\nunverified guesses (review before trusting):")
        for r in guesses:
            print(f"  {r.name:30} {r.board}:{r.slug:20} {r.website}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="liveness-check BACKFILL rows instead")
    ap.add_argument("--report", action="store_true", help="print the table from the DB and exit")
    ap.add_argument("--region", choices=["AU", "GULF", "REMOTE"], help="only probe this region")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.report:
        await report()
    else:
        await run_probe(args.workers, args.backfill, args.region)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
