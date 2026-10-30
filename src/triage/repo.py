"""The database calls graph.py needs, so the graph never touches SQL.

Each function opens its own session and commits, and is safe to call twice: a node can re-run when the graph
resumes after the interrupt.
"""
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from triage.db import SessionLocal
from triage.models import Company, Post, TriageRun

STATUSES = {"new", "duplicate", "awaiting_review", "resuming", "shortlisted", "rejected", "expired"}
_RUN_FIELDS = ("jev_p", "hi", "lo", "route", "deep_read", "final_status", "decided_by", "decision", "latency_ms", "error", "reason")


async def get_post_dict(post_id: int) -> dict | None:
    """One post as a plain dict (str/int/float/None only) for the graph state, or None if it doesn't exist."""
    async with SessionLocal() as s:
        row = (await s.execute(select(Post, Company.name).join(Company, Company.id == Post.company_id)
                               .where(Post.id == post_id))).first()
    if row is None:
        return None
    p, company = row
    return {"id": p.id, "version": p.version, "company": company, "title": p.title, "location": p.location,
            "url": p.url, "description_text": p.description_text, "remote_eligibility": p.remote_eligibility,
            "status": p.status, "closed": p.closed_at is not None, "duplicate_of": p.duplicate_of}


async def set_status(post_id: int, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {sorted(STATUSES)}")
    async with SessionLocal() as s, s.begin():
        (await s.get(Post, post_id)).status = status


async def record_run(*, post_id: int, post_version: int, criteria_version: str, **fields) -> int:
    """Insert the run row for (post, post version, criteria version), or update it if it already exists.
    Only the fields you pass are written, so a later call can add the decision without wiping the screen result."""
    unknown = set(fields) - set(_RUN_FIELDS)
    if unknown:
        raise TypeError(f"unknown run fields: {sorted(unknown)}")
    stmt = insert(TriageRun).values(post_id=post_id, post_version=post_version, criteria_version=criteria_version, **fields)
    if fields:
        stmt = stmt.on_conflict_do_update(constraint="uq_runs_post_version_criteria", set_=fields)
    else:
        stmt = stmt.on_conflict_do_update(constraint="uq_runs_post_version_criteria", set_={"post_id": post_id})
    async with SessionLocal() as s, s.begin():
        return (await s.execute(stmt.returning(TriageRun.id))).scalar_one()


async def clear_error(post_id: int, post_version: int, criteria_version: str) -> None:
    """A later success wipes an earlier failure from the same run row (record_run skips None, so it can't)."""
    async with SessionLocal() as s, s.begin():
        await s.execute(update(TriageRun).where(
            TriageRun.post_id == post_id, TriageRun.post_version == post_version,
            TriageRun.criteria_version == criteria_version).values(error=None))
