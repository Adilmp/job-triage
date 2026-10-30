r"""The triage graph: one post in, a shortlist proposal or a rejection out, with a human in the loop.

    check_rules -> screen_with_jev -> pick_route -+-> (score high)  ------------------------+
         |                                        +-> (score low)   -> record_result        |
         +-> (dropped) -> record_result          +-> read_closely --(fit / unsure)---------+--> ask_human -> record_result
                                                                    \--(sure it's no)--> record_result

State holds plain values only (str/int/float/dict/list), so the Postgres checkpointer stores it with no custom
serializer. One thread per post: thread_id = str(post_id).

Resume rules (LangGraph restarts the interrupted node from its first line when it resumes):
  - only `ask_human` runs again, and before its interrupt() it does two idempotent writes (set_status, an upserted run row)
  - Jev and Ollama are never called twice: their results live in state, and each node skips work already done
  - hi / lo / min_conf are copied into state once, in pick_route, and never re-read, so editing config
    while a post waits for review can't change how it is judged
  - the final status comes from the final state, never from the request that resumed the graph
"""
import functools
import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from triage import repo
from triage.config import settings
from triage.criteria import criteria_version
from triage.services import jev
from triage.services.deep_read import DeepRead, deep_read
from triage.services.prefilter import prefilter

log = logging.getLogger(__name__)


class TriageState(TypedDict, total=False):
    post_id: int
    post: dict
    criteria_version: str
    reason: str            # why the prefilter dropped it
    jev_p: float
    hi: float
    lo: float
    min_conf: float
    route: str             # prefilter_drop | auto_reject | shortlist | deep_read
    deep_read: dict
    proposed: bool         # True if a human must decide
    why: str               # why it was proposed: jev_high | deep_read_fit | deep_read_unsure_reject
    decision: str          # approve | reject (only set when a human decided)
    note: str
    status: str            # final: shortlisted | rejected
    latency_ms: int        # compute time only; the wait for a human isn't counted


# ---------- pure decisions (unit-tested on their own) ----------

def screen_route(jev_p: float, hi: float, lo: float) -> str:
    if jev_p >= hi:
        return "shortlist"
    if jev_p <= lo:
        return "auto_reject"
    return "deep_read"


def needs_human(dr: DeepRead, min_conf: float) -> tuple[bool, str]:
    """A fit always goes to the human. A 'not fit' goes too unless the model is confident, so shaky rejections
    aren't silently lost."""
    if dr.fit:
        return True, "deep_read_fit"
    if dr.confidence < min_conf:
        return True, "deep_read_unsure_reject"
    return False, "deep_read_reject"


def parse_decision(answer) -> tuple[str, str]:
    """(decision, note) from whatever the resume carried. Anything that isn't a clear 'approve' is a reject."""
    note = ""
    if isinstance(answer, dict):
        note = str(answer.get("note") or "")
        answer = answer.get("decision")
    if isinstance(answer, str) and answer.strip().lower() == "approve":
        return "approve", note
    if not (isinstance(answer, str) and answer.strip().lower() == "reject"):
        log.error("malformed resume value %r; treating as reject", answer)
    return "reject", note


# ---------- nodes ----------

def _timed(node):
    @functools.wraps(node)
    async def wrapper(state):
        start = time.monotonic()
        update = dict(await node(state) or {})
        update["latency_ms"] = int(state.get("latency_ms", 0) + (time.monotonic() - start) * 1000)
        return update
    return wrapper


async def _save_run(state: TriageState, **extra) -> None:
    fields = {k: state.get(k) for k in ("jev_p", "hi", "lo", "route", "deep_read", "reason")}
    fields.update(extra)
    await repo.record_run(post_id=state["post_id"], post_version=state["post"]["version"],
                          criteria_version=state["criteria_version"], **{k: v for k, v in fields.items() if v is not None})


@_timed
async def check_rules(state: TriageState) -> dict:
    post = state.get("post") or await repo.get_post_dict(state["post_id"])
    if post is None:
        raise ValueError(f"post {state['post_id']} not found")
    update = {"post": post, "criteria_version": state.get("criteria_version") or criteria_version()}
    if (result := prefilter(post)).drop:
        update |= {"route": "prefilter_drop", "reason": result.reason}
    return update


@_timed
async def screen_with_jev(state: TriageState) -> dict:
    if state.get("jev_p") is not None:      # a replay must not pay for the same answer twice
        return {}
    probs = await jev.ask(jev.post_text(state["post"]))
    return {"jev_p": probs["fit"]}


@_timed
async def pick_route(state: TriageState) -> dict:
    hi = state["hi"] if state.get("hi") is not None else settings.HI
    lo = state["lo"] if state.get("lo") is not None else settings.LO
    min_conf = state["min_conf"] if state.get("min_conf") is not None else settings.DEEP_READ_MIN_CONFIDENCE
    route = screen_route(state["jev_p"], hi, lo)
    update = {"hi": hi, "lo": lo, "min_conf": min_conf, "route": route}
    if route == "shortlist":
        update |= {"proposed": True, "why": "jev_high"}
    return update


@_timed
async def read_closely(state: TriageState) -> dict:
    dump = state.get("deep_read")
    if dump is None:
        dump = (await deep_read(state["post"])).model_dump()
    proposed, why = needs_human(DeepRead(**dump), state["min_conf"])
    return {"deep_read": dump, "proposed": proposed, "why": why}


async def ask_human(state: TriageState) -> dict:
    post = state["post"]
    await repo.set_status(state["post_id"], "awaiting_review")      # idempotent: this node reruns on resume
    await _save_run(state)                                           # so the review queue can show the score and reasons
    answer = interrupt({
        "kind": "approval_request", "post_id": state["post_id"], "title": post["title"], "company": post["company"],
        "location": post["location"], "url": post["url"], "jev_p": state["jev_p"], "hi": state["hi"], "lo": state["lo"],
        "route": state["route"], "why": state["why"], "deep_read": state.get("deep_read"),
    })
    decision, note = parse_decision(answer)
    return {"decision": decision, "note": note}


@_timed
async def record_result(state: TriageState) -> dict:
    decision = state.get("decision")
    if decision is not None:
        status, decided_by = ("shortlisted" if decision == "approve" else "rejected"), "human"
    else:
        status, decided_by = "rejected", "auto"
    await _save_run(state, final_status=status, decided_by=decided_by, decision=decision,
                    latency_ms=state.get("latency_ms"))
    await repo.set_status(state["post_id"], status)
    return {"status": status}


# ---------- wiring ----------

async def build_graph(checkpointer):
    g = StateGraph(TriageState)
    g.add_node("check_rules", check_rules)
    g.add_node("screen_with_jev", screen_with_jev)
    g.add_node("pick_route", pick_route)
    g.add_node("read_closely", read_closely)
    g.add_node("ask_human", ask_human)
    g.add_node("record_result", record_result)

    g.add_edge(START, "check_rules")
    g.add_conditional_edges("check_rules", lambda s: "record_result" if s.get("route") == "prefilter_drop" else "screen_with_jev",
                            ["record_result", "screen_with_jev"])
    g.add_edge("screen_with_jev", "pick_route")
    g.add_conditional_edges(
        "pick_route", lambda s: {"shortlist": "ask_human", "auto_reject": "record_result", "deep_read": "read_closely"}[s["route"]],
        ["ask_human", "record_result", "read_closely"])
    g.add_conditional_edges("read_closely", lambda s: "ask_human" if s["proposed"] else "record_result", ["ask_human", "record_result"])
    g.add_edge("ask_human", "record_result")
    g.add_edge("record_result", END)
    return g.compile(checkpointer=checkpointer)
