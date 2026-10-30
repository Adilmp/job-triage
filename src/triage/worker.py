import logging
import random
import time
from datetime import datetime, timedelta, timezone

from arq import Retry, cron
from arq.connections import RedisSettings
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select, update

from triage import repo
from triage.config import settings
from triage.criteria import criteria_version
from triage.db import SessionLocal, engine
from triage.errors import PermanentError, RetryableError
from triage.fetchers import FETCHERS
from triage.fetchers.base import make_client
from triage.graph import build_graph
from triage.ingest import ingest
from triage.models import Company, Post, TriageRun

log = logging.getLogger(__name__)

MAX_TRIES = 5
REVIEW_EXPIRY = timedelta(days=7)
FETCH_WINDOW = 6 * 3600          # seconds; also the cron period
LONG_WAIT = 3600                 # a Retry-After longer than this isn't worth holding a job for
_SITES = {"greenhouse": "greenhouse.io", "lever": "lever.co", "ashby": "ashbyhq.com", "recruitee": "recruitee.com",
          "workable": "workable.com", "teamtailor": "teamtailor.com"}


def triage_job_id(post_id: int, version: int) -> str:
    """Post + version + criteria: enqueueing the same work twice is a no-op, while a criteria change is new work."""
    return f"triage:{post_id}:{version}:{criteria_version()}"


def _retry_delay(job_try: int, retry_after: int | None = None) -> float:
    backoff = 2 ** job_try * 10                                  # 20s, 40s, 80s, 160s
    return max(retry_after or 0, backoff * (1 + random.random() * 0.1))


def _site(company: Company) -> str:
    # Teamtailor career sites on a company's own domain are their own site.
    return company.slug if company.board == "teamtailor" and "." in company.slug else _SITES[company.board]


async def _record_error(ctx, company_id: int, message: str) -> dict:
    async with ctx["db"]() as s, s.begin():
        (await s.get(Company, company_id)).last_error = message[:500]
    return {"error": message}


async def fetch_company(ctx, company_id: int) -> dict:
    async with ctx["db"]() as s:
        company = await s.get(Company, company_id)
    if company is None or not company.verified or not company.board:
        return {"skipped": "unverified or no board"}

    site, redis = _site(company), ctx["redis"]
    if await redis.get(f"cooldown:{site}"):
        # A site that asked us to wait (Workable: ~24h after a burst) is left alone, so we don't extend the ban.
        return {"skipped": f"{site} cooldown"}

    try:
        posts = await FETCHERS[company.board](ctx["http"], company.slug)
    except PermanentError as e:
        return await _record_error(ctx, company_id, f"{company.board}:{company.slug} {e}")
    except RetryableError as e:
        if e.retry_after:
            await redis.set(f"cooldown:{site}", "1", ex=e.retry_after)
            if e.retry_after > LONG_WAIT:
                return await _record_error(ctx, company_id, f"{site} rate-limited us for {e.retry_after}s: {e}")
        if ctx["job_try"] >= MAX_TRIES:
            return await _record_error(ctx, company_id, f"gave up after {MAX_TRIES} tries: {e}")
        raise Retry(defer=_retry_delay(ctx["job_try"], e.retry_after)) from e

    async with ctx["db"]() as s, s.begin():
        res = await ingest(s, await s.get(Company, company_id), posts)
    if settings.TRIAGE_ENABLED:
        for post_id, version in res.to_triage:
            await redis.enqueue_job("triage_post", post_id, version, _job_id=triage_job_id(post_id, version))
    log.info("fetched %s: %s", company.name, res)
    return {"new": res.new, "changed": res.changed, "unchanged": res.unchanged, "closed": res.closed,
            "duplicates": res.duplicates, "dropped_empty": res.dropped_empty}


async def _drop_thread(ctx, post_id: int) -> None:
    """P11: the checkpoint tables only hold posts that are still in flight. triage_runs is the audit trail."""
    await ctx["saver"].adelete_thread(str(post_id))


async def _pending_interrupts(ctx, cfg) -> list:
    state = await ctx["graph"].aget_state(cfg)
    return [i for t in state.tasks for i in t.interrupts]


async def triage_post(ctx, post_id: int, version: int) -> dict:
    graph, cv = ctx["graph"], criteria_version()
    cfg = {"configurable": {"thread_id": str(post_id)}}
    post = await repo.get_post_dict(post_id)
    if post is None or post["version"] != version:
        return {"skipped": "stale version"}                      # the post was edited again: a newer job owns it
    if post["closed"]:
        return {"skipped": "closed"}
    if post["duplicate_of"]:
        return {"skipped": "duplicate"}

    state = await graph.aget_state(cfg)
    values = state.values or {}
    same_run = bool(values) and values["post"]["version"] == version and values["criteria_version"] == cv
    if same_run and state.next:
        if await _pending_interrupts(ctx, cfg):
            return {"status": "awaiting_review"}                 # already paused for exactly this work
        graph_input = None                                       # an earlier attempt died mid-graph: continue from its checkpoint
    else:
        if values:                                               # left over from an older version, other criteria, or a half-cleaned finish
            await _drop_thread(ctx, post_id)
        if same_run:
            return {"skipped": "already done"}
        graph_input = {"post_id": post_id}

    try:
        out = await graph.ainvoke(graph_input, cfg)
    except RetryableError as e:
        if ctx["job_try"] < MAX_TRIES:
            raise Retry(defer=_retry_delay(ctx["job_try"], e.retry_after)) from e
        await repo.record_run(post_id=post_id, post_version=version, criteria_version=cv, error=f"gave up after {MAX_TRIES} tries: {e}"[:500])
        return {"error": str(e)}
    except PermanentError as e:
        await repo.record_run(post_id=post_id, post_version=version, criteria_version=cv, error=str(e)[:500])
        return {"error": str(e)}

    await repo.clear_error(post_id, version, cv)
    if out.get("__interrupt__"):
        return {"status": "awaiting_review"}
    await _drop_thread(ctx, post_id)
    return {"status": out["status"]}


async def resume_triage(ctx, post_id: int, decision: str, note: str | None = None) -> dict:
    """Runs after the API claimed the post (awaiting_review -> resuming, P9). The final status comes from the
    graph's final state, not from the request (P10)."""
    graph = ctx["graph"]
    cfg = {"configurable": {"thread_id": str(post_id)}}
    pending = await _pending_interrupts(ctx, cfg)
    if not pending:
        log.error("resume_triage(%s): no pending approval; sending the post back to be triaged again", post_id)
        await repo.set_status(post_id, "new")
        return {"error": "no pending approval"}
    kind = pending[0].value.get("kind") if isinstance(pending[0].value, dict) else None
    if kind != "approval_request":
        log.error("resume_triage(%s): unknown interrupt kind %r; not resuming", post_id, kind)
        await repo.set_status(post_id, "awaiting_review")
        raise ValueError(f"unknown interrupt kind {kind!r} for post {post_id}")

    try:
        out = await graph.ainvoke(Command(resume={"decision": decision, "note": note or ""}), cfg)
    except RetryableError as e:
        if ctx["job_try"] < MAX_TRIES:
            raise Retry(defer=_retry_delay(ctx["job_try"], e.retry_after)) from e
        await repo.set_status(post_id, "awaiting_review")        # give it back so the decision can be made again
        return {"error": str(e)}
    except PermanentError as e:
        await repo.set_status(post_id, "awaiting_review")
        return {"error": str(e)}

    if out.get("__interrupt__") or "status" not in out:
        log.error("resume_triage(%s): graph did not finish: %s", post_id, out)
        await repo.set_status(post_id, "awaiting_review")
        return {"error": "graph did not finish"}
    await _drop_thread(ctx, post_id)
    return {"status": out["status"]}


async def expire_stale_reviews(ctx) -> int:
    """P12: a review nobody answered for 7 days is 'expired'. No label is written: a timeout isn't a "no", and
    a fake label would poison the gold data."""
    cutoff = datetime.now(timezone.utc) - REVIEW_EXPIRY
    async with ctx["db"]() as s, s.begin():
        stale = select(TriageRun.post_id).join(Post, Post.id == TriageRun.post_id).where(
            TriageRun.post_version == Post.version, TriageRun.final_status.is_(None), TriageRun.created_at < cutoff)
        # the status condition makes this atomic against the API claiming the same post
        ids = (await s.scalars(update(Post).where(Post.status == "awaiting_review", Post.id.in_(stale))
                               .values(status="expired").returning(Post.id))).all()
        if ids:
            await s.execute(update(TriageRun).where(TriageRun.post_id.in_(ids), TriageRun.final_status.is_(None))
                            .values(final_status="expired"))
    for post_id in ids:
        await _drop_thread(ctx, post_id)
    return len(ids)


async def fetch_all(ctx) -> int:
    window = int(time.time() // FETCH_WINDOW)
    async with ctx["db"]() as s:
        ids = (await s.scalars(select(Company.id).where(Company.verified, Company.board.is_not(None)).order_by(Company.id))).all()
    for cid in ids:
        # overlapping crons in one window get the same job id, so they enqueue nothing
        await ctx["redis"].enqueue_job("fetch_company", cid, _job_id=f"fetch:{cid}:{window}")
    return len(ids)


async def on_startup(ctx) -> None:
    ctx["http"], ctx["db"] = make_client(), SessionLocal
    # The worker owns the LangGraph checkpointer; the graph is built once and shared by the job slots.
    ctx["pool"] = AsyncConnectionPool(settings.psycopg_url, max_size=10, open=False,
                                      kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await ctx["pool"].open()
    ctx["saver"] = AsyncPostgresSaver(ctx["pool"])
    await ctx["saver"].setup()
    ctx["graph"] = await build_graph(ctx["saver"])


async def on_shutdown(ctx) -> None:
    await ctx["http"].aclose()
    await ctx["pool"].close()
    await engine.dispose()


class WorkerSettings:
    functions = [fetch_company, fetch_all, triage_post, resume_triage, expire_stale_reviews]
    cron_jobs = [cron(fetch_all, hour={0, 6, 12, 18}, minute=0), cron(expire_stale_reviews, hour=3, minute=0)]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_tries = MAX_TRIES
    max_jobs = 4
    job_timeout = 600          # covers an Ollama cold start later on
