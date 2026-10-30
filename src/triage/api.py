"""HTTP API. curl and /docs are the UI.

The API never runs the graph: it reads the database and enqueues jobs. The worker owns the graph and its
checkpointer.
"""
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert

from triage import queries
from triage.config import settings
from triage.criteria import criteria_version
from triage.db import SessionLocal, engine
from triage.models import Company, Label, Post, TriageRun
from triage.worker import triage_job_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = SessionLocal
    app.state.redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    yield
    await app.state.redis.aclose()
    await engine.dispose()


app = FastAPI(title="job-triage", lifespan=lifespan)

ListStatus = Literal["new", "awaiting_review", "resuming", "shortlisted", "rejected", "expired", "duplicate"]
Region = Literal["AU", "GULF", "REMOTE", "BACKFILL"]


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = None


def _db(request: Request):
    return request.app.state.db()


def _run_dict(r: TriageRun) -> dict:
    return {"id": r.id, "post_version": r.post_version, "criteria_version": r.criteria_version, "jev_p": r.jev_p,
            "hi": r.hi, "lo": r.lo, "route": r.route, "deep_read": r.deep_read, "reason": r.reason,
            "final_status": r.final_status, "decided_by": r.decided_by, "decision": r.decision,
            "latency_ms": r.latency_ms, "error": r.error, "created_at": r.created_at.isoformat() if r.created_at else None}


@app.get("/health")
async def health(request: Request):
    out = {"db": "ok", "redis": "ok", "ollama": "ok"}
    try:
        async with _db(request) as s:
            await s.execute(text("SELECT 1"))
    except Exception:
        out["db"] = "down"
    try:
        await request.app.state.redis.ping()
    except Exception:
        out["redis"] = "down"
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            (await c.get(f"{settings.OLLAMA_URL}/api/tags")).raise_for_status()
    except Exception:
        out["ollama"] = "down"
    # Without the DB or Redis nothing works. Without Ollama only the deep read waits, so that is "degraded".
    if "down" in (out["db"], out["redis"]):
        return JSONResponse({"status": "down", **out}, status_code=503)
    return {"status": "degraded" if out["ollama"] == "down" else "ok", **out}


@app.post("/companies/{company_id}/fetch", status_code=202)
async def fetch_now(company_id: int, request: Request):
    async with _db(request) as s:
        if await s.get(Company, company_id) is None:
            raise HTTPException(404, "company not found")
    await request.app.state.redis.enqueue_job("fetch_company", company_id, _job_id=f"fetch:{company_id}:manual:{time.time_ns()}")
    return {"queued": True}


@app.get("/posts")
async def list_posts(request: Request, status: ListStatus = "awaiting_review", region: Region | None = None,
                     limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """The review queue (and other lists), each post with its latest run's score and deep-read reasons."""
    async with _db(request) as s:
        q = (select(Post, Company.name, Company.region).join(Company, Company.id == Post.company_id)
             .where(Post.status == status).order_by(Post.id).limit(limit).offset(offset))
        if region:
            q = q.where(Company.region == region)
        rows = (await s.execute(q)).all()
        ids = [p.id for p, _, _ in rows]
        latest: dict[int, TriageRun] = {}
        for r in (await s.scalars(select(TriageRun).where(TriageRun.post_id.in_(ids)).order_by(TriageRun.id))).all():
            latest[r.post_id] = r                                   # last one wins: the newest run
    items = []
    for p, company, reg in rows:
        r = latest.get(p.id)
        items.append({"id": p.id, "title": p.title, "company": company, "region": reg, "location": p.location,
                      "url": p.url, "status": p.status, "remote_eligibility": p.remote_eligibility,
                      "jev_p": r.jev_p if r else None, "route": r.route if r else None,
                      "deep_read": r.deep_read if r else None})
    return {"items": items, "count": len(items)}


@app.get("/posts/{post_id}")
async def get_post(post_id: int, request: Request):
    async with _db(request) as s:
        row = (await s.execute(select(Post, Company.name).join(Company, Company.id == Post.company_id)
                               .where(Post.id == post_id))).first()
        if row is None:
            raise HTTPException(404, "post not found")
        p, company = row
        runs = (await s.scalars(select(TriageRun).where(TriageRun.post_id == post_id).order_by(TriageRun.id.desc()))).all()
    return {"id": p.id, "title": p.title, "company": company, "location": p.location, "url": p.url, "status": p.status,
            "version": p.version, "remote_eligibility": p.remote_eligibility, "comp_min": p.comp_min,
            "comp_max": p.comp_max, "comp_currency": p.comp_currency, "description_text": p.description_text,
            "runs": [_run_dict(r) for r in runs]}


@app.post("/posts/{post_id}/decision")
async def decide(post_id: int, body: DecisionIn, request: Request):
    # P9: the claim is one conditional UPDATE. Two clicks, or a click racing the expiry cron: exactly one wins.
    async with _db(request) as s, s.begin():
        claimed = (await s.execute(update(Post).where(Post.id == post_id, Post.status == "awaiting_review")
                                   .values(status="resuming").returning(Post.version))).first()
        if claimed is None:
            post = await s.get(Post, post_id)
            if post is None:
                raise HTTPException(404, "post not found")
            raise HTTPException(409, f"post is '{post.status}', not awaiting_review")
        stmt = insert(Label).values(post_id=post_id, source="approval", label="fit" if body.decision == "approve" else "not_fit",
                                    notes=body.note)
        await s.execute(stmt.on_conflict_do_update(index_elements=[Label.post_id, Label.source],
                                                   set_={"label": stmt.excluded.label, "notes": stmt.excluded.notes}))
    try:
        job = await request.app.state.redis.enqueue_job("resume_triage", post_id, body.decision, body.note,
                                                        _job_id=f"resume:{post_id}:{claimed.version}:{time.time_ns()}")
        if job is None:
            raise RuntimeError("job was not enqueued")
    except Exception as e:
        # Undo the claim so the decision can be sent again, instead of leaving the post stuck in 'resuming'.
        async with _db(request) as s, s.begin():
            await s.execute(update(Post).where(Post.id == post_id, Post.status == "resuming").values(status="awaiting_review"))
            await s.execute(delete(Label).where(Label.post_id == post_id, Label.source == "approval"))
        raise HTTPException(503, f"could not queue the decision: {e}") from e
    return {"post_id": post_id, "status": "resuming", "decision": body.decision}


@app.post("/retriage")
async def retriage(request: Request, stale: bool = True, limit: int = Query(50, ge=1, le=1000)):
    """P7. stale=true: only posts last judged under older criteria. stale=false: also posts never triaged.
    `limit` is capped: the backlog is thousands of posts, and each middle-band one costs a minute of local LLM."""
    cv = criteria_version()
    async with _db(request) as s:
        todo = await queries.list_untriaged(s, cv, limit, stale_only=stale)
    for post_id, version in todo:
        await request.app.state.redis.enqueue_job("triage_post", post_id, version, _job_id=triage_job_id(post_id, version))
    return {"queued": len(todo), "criteria_version": cv}


@app.get("/stats")
async def stats(request: Request):
    async with _db(request) as s:
        return await queries.stats(s)
