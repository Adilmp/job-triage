# job-triage

A backend that pulls job posts from company career boards, screens them cheaply, has a local LLM read the
uncertain ones closely, and stops for a human before anything is shortlisted. Built to find early-career AI and
backend roles that can be done remotely from Pakistan, or from an office there.

**Status:** the pipeline, worker, API and eval harness are built and tested (270 tests). The held-out evaluation
has not been run yet; see [Evaluation](#evaluation).

## How a post flows

```
career boards ──► fetch (6 ATS APIs) ──► normalize + dedupe ──► Postgres
                                                                   │
                ┌──────────────────────── triage graph (LangGraph) ┘
                ▼
   prefilter (rules, free) ──drop──► rejected
        │
   Jev screen (one API call, ~$0.00002) ──score ≥ HI──► propose ──┐
        │                              └─score ≤ LO──► rejected   │
        └─ in between ─► deep read (local qwen2.5:7b) ──fit / unsure──┤
                                        └─ confident "no" ─► rejected
                                                                   ▼
                                      human approval (interrupt, survives restarts)
                                                                   ▼
                                                     shortlisted / rejected
```

- **Fetchers:** Greenhouse, Lever, Ashby, Recruitee, Workable and Teamtailor (RSS). Rate limits are honoured:
  a site that asks us to wait (Workable asks for 24 hours) is skipped until the wait is over.
- **Deterministic first:** remote eligibility and salary come from each board's typed fields, never from a model.
  When the fields can't settle it, the answer is `unclear`, not a guess.
- **Two-stage screening:** every post gets the cheap Jev call; only the uncertain middle band pays for a slow
  local-LLM read. A "not fit" the LLM isn't confident about still goes to the human.
- **Human in the loop:** posts proposed for the shortlist pause in a LangGraph `interrupt()`. State lives in a
  Postgres checkpointer, so a pause survives a worker restart. Decisions go through an atomic claim, so two clicks
  can't resume the same post twice.

## Stack

Python 3.12, FastAPI, arq (Redis), SQLAlchemy + Alembic on Postgres 16, LangGraph with `AsyncPostgresSaver`,
LangChain `ChatOllama` with structured output, httpx, pytest with testcontainers. Everything runs locally for $0
apart from the Jev calls.

## Run it

```bash
cp .env.example .env            # add JEV_API_KEY
docker compose up -d            # postgres, redis, api (port 8010), worker
uv run alembic upgrade head
OLLAMA_KEEP_ALIVE=24h ollama serve   # on the host; the api and worker reach it on 127.0.0.1
curl localhost:8010/health
```

`api` and `worker` use host networking so they can reach Ollama on `127.0.0.1` without exposing Ollama to the
network. `triage.seed` loads company lists from local research CSVs that aren't part of this repo;
`data/remote_first.csv` is included.

API (`/docs` has the full schema):

| | |
|---|---|
| `GET /posts?status=awaiting_review` | the review queue, with each post's score and deep-read reasons |
| `POST /posts/{id}/decision` | `{"decision": "approve" \| "reject", "note": ...}`; 409 if someone already decided |
| `GET /posts/{id}` | a post and its triage history |
| `POST /retriage?stale=true` | re-run posts judged under older criteria |
| `GET /stats` | counts by route and status, p50/p95 latency |

## Evaluation

250 posts labeled by hand (`evals/gold.csv`) with a terminal labeler that never shows a model's score. Labels are
fit, stretch (senior on paper but worth applying) or not fit. `evals/split.csv` freezes a stratified split:
149 posts to choose the thresholds on, 101 held out and run once. `evals/run_eval.py` sends each post through the
real graph, lets the gold label answer the human step, and reports every rate with a 95% Wilson interval: Jev's
AUC, the false-reject rate by stage, deep-read accuracy against Jev alone, shortlist precision and recall, review
load, latency and cost. A regression gate compares each run with a saved baseline.

**Found so far, on the tuning split only:**
- Jev ranks fits above non-fits reasonably (AUC 0.77) but its scores are compressed near zero: the median fit
  scores 0.04 and the median non-fit 0.02. The default auto-reject line of 0.2 would have thrown away about 80% of
  real fits. ([known issue #2](docs/KNOWN_ISSUES.md))
- The seniority rule dropped "Member of Technical Staff", an ordinary engineering title at AI labs. Fixed, with a
  regression test. ([known issue #1](docs/KNOWN_ISSUES.md))

Held-out numbers will be added here after the thresholds are chosen.

## More

- [docs/DESIGN.md](docs/DESIGN.md): the design patterns (P1 to P16) and the prior art they came from
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): every failure found, with evidence
- [docs/FIT.md](docs/FIT.md): what "fit" means for the labels

## Limitations

- Teamtailor has no JSON API, so it is read from each career site's RSS feed (best effort).
- Lever boards hosted in the EU (`api.eu.lever.co`) aren't supported.
- With ~20 positives in the held-out split, every number has a wide confidence interval.

## Prior art

[akminx/compass](https://github.com/akminx/compass) and [luisvid/JobGraph](https://github.com/luisvid/JobGraph).
Several patterns here came from reading their code; [docs/DESIGN.md](docs/DESIGN.md) lists which.
