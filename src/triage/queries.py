"""SQL the API needs that isn't a single-row lookup."""
from sqlalchemy import or_, select, text

from triage.models import Post, TriageRun


async def list_untriaged(session, cv: str, limit: int, stale_only: bool = False) -> list[tuple[int, int]]:
    """(post_id, version) of open, non-duplicate posts with no finished (or currently paused) run for their
    current version under criteria `cv` (P7). stale_only: just those that were judged under older criteria."""
    current = select(TriageRun.id).where(
        TriageRun.post_id == Post.id, TriageRun.post_version == Post.version, TriageRun.criteria_version == cv,
        or_(TriageRun.final_status.is_not(None), Post.status == "awaiting_review"))
    q = (select(Post.id, Post.version)
         .where(Post.closed_at.is_(None), Post.duplicate_of.is_(None), Post.status != "resuming", ~current.exists())
         .order_by(Post.id).limit(limit))
    if stale_only:
        q = q.where(select(TriageRun.id).where(
            TriageRun.post_id == Post.id, TriageRun.post_version == Post.version, TriageRun.criteria_version != cv).exists())
    return [tuple(r) for r in (await session.execute(q)).all()]


async def stats(session) -> dict:
    posts = dict((await session.execute(text("SELECT status, count(*) FROM posts GROUP BY status"))).all())
    routes = dict((await session.execute(text(
        "SELECT route, count(*) FROM triage_runs WHERE route IS NOT NULL GROUP BY route"))).all())
    decided = dict((await session.execute(text(
        "SELECT decided_by, count(*) FROM triage_runs WHERE decided_by IS NOT NULL GROUP BY decided_by"))).all())
    p50, p95, n = (await session.execute(text(
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), count(latency_ms) FROM triage_runs"))).one()
    return {"posts_by_status": posts, "runs_by_route": routes, "decided_by": decided,
            "latency_ms": {"p50": p50, "p95": p95, "n": n}}
