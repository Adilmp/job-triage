"""Turn one company's fetched posts into `posts` rows.

Identity is (provider, external_id). Per fetch:
  new id                    -> insert, queue triage (unless it duplicates an open post from another board)
  same id, same hash        -> touch last_seen_at (and reopen if it had closed)
  same id, different hash   -> version += 1, update, status back to 'new', queue re-triage
  open post not in fetch    -> closed_at = now
  empty description         -> never inserted or updated; logged and counted in companies.last_error (P2)
The caller commits.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, update

from triage.derive import derive
from triage.fetchers.base import RawPost
from triage.models import Company, Post
from triage.normalize import content_hash, fingerprint

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    closed: int = 0
    duplicates: int = 0
    dropped_empty: int = 0
    to_triage: list[tuple[int, int]] = field(default_factory=list)   # (post_id, version) for the worker to enqueue


def _fields(company: Company, p: RawPost) -> dict:
    d = derive(p.provider, p.raw, p.location)
    return {
        "url": p.url, "title": p.title, "location": p.location, "description_text": p.description_text, "raw": p.raw,
        "remote_eligibility": d.remote_eligibility, "comp_min": d.comp_min, "comp_max": d.comp_max,
        "comp_currency": d.comp_currency,
        "fingerprint": fingerprint(company.name, p.title),
        "content_hash": content_hash(p.title, p.location, p.description_text),
    }


async def ingest(session, company: Company, posts: list[RawPost], provider: str | None = None) -> IngestResult:
    """`provider` is the board that was fetched (defaults to the company's); only its posts can be closed."""
    provider = provider or company.board
    now = datetime.now(timezone.utc)
    res = IngestResult()
    seen: set[str] = set()
    existing = {
        p.external_id: p
        for p in await session.scalars(select(Post).where(Post.company_id == company.id, Post.provider == provider))
    }

    for rp in posts:
        seen.add(rp.external_id)         # a dropped post was still on the board: don't close it
        if not rp.description_text.strip():
            res.dropped_empty += 1
            log.warning("dropped %s:%s %r from %s: empty description", rp.provider, rp.external_id, rp.title, company.name)
            continue
        f = _fields(company, rp)
        row = existing.get(rp.external_id)

        if row is None:
            primary = await session.scalar(
                select(Post.id).where(Post.fingerprint == f["fingerprint"], Post.provider != rp.provider,
                                      Post.closed_at.is_(None), Post.duplicate_of.is_(None)).order_by(Post.id).limit(1))
            row = Post(company_id=company.id, provider=rp.provider, external_id=rp.external_id, version=1,
                       status="duplicate" if primary else "new", duplicate_of=primary,
                       first_seen_at=now, last_seen_at=now, **f)
            session.add(row)
            await session.flush()
            if primary:
                res.duplicates += 1
            else:
                res.new += 1
                res.to_triage.append((row.id, 1))
        elif row.content_hash == f["content_hash"]:
            row.last_seen_at, row.closed_at = now, None
            res.unchanged += 1
        else:
            for k, v in f.items():
                setattr(row, k, v)
            row.version += 1
            row.last_seen_at, row.closed_at = now, None
            res.changed += 1
            if row.duplicate_of is None:
                row.status = "new"
                res.to_triage.append((row.id, row.version))

    stale = update(Post).where(Post.company_id == company.id, Post.provider == provider, Post.closed_at.is_(None))
    if seen:
        stale = stale.where(Post.external_id.not_in(seen))
    res.closed = (await session.execute(stale.values(closed_at=now))).rowcount

    company.last_fetched_at = now
    company.last_error = (f"{res.dropped_empty} posts dropped: empty description" if res.dropped_empty else None)
    return res
