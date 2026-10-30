# Design notes

Code comments refer to these patterns as P1 to P16. Most were borrowed, after reading their source, from two
open-source projects: [akminx/compass](https://github.com/akminx/compass) (Greenhouse/Lever/Ashby into a LangGraph
pipeline with interrupt-based approval) and [luisvid/JobGraph](https://github.com/luisvid/JobGraph) (Postgres +
LangGraph job scoring with ADRs and fingerprint dedupe).

| # | Pattern | From | Why |
|---|---|---|---|
| P1 | Greenhouse needs `?content=true`, and its `content` is HTML-escaped: unescape, drop `<script>`/`<style>`, strip tags, tidy whitespace | compass | Without it a model only sees the title |
| P2 | Posts with an empty description are dropped loudly (logged and counted), never passed on | compass | An empty description makes an LLM invent a job from its title |
| P3 | Ashby quirks: `descriptionPlain` with an HTML fallback, `location` or `locationName`, compensation only when the board says to display it | compass | Real field quirks |
| P4 | Keep the raw ATS payload and derive remote/compensation deterministically from typed fields before any model sees the post | JobGraph | Free and reliable; LLMs mark city-based roles as remote |
| P5 | `remote_eligibility` is one of eligible, region_locked, hybrid, onsite, unclear. Only onsite/hybrid outside the target locations is rejected; `unclear` goes through | JobGraph, compass | A wrong yes/no costs more than "don't know" |
| P6 | Two hashes: a cross-board fingerprint (normalized company + title) for "same role", and a content hash for "edited post" | JobGraph | Two different questions |
| P7 | `criteria_version` (a hash of the Jev wording, prefilter rules, deep-read prompt and thresholds) on every triage run | JobGraph, compass | Changing the criteria re-triages old posts instead of leaving stale verdicts |
| P8 | Sticky thresholds: HI/LO are copied into graph state at route time | compass | Editing config while a post waits for review can't change its outcome |
| P9 | Atomic claim before resume: one conditional `UPDATE ... WHERE status='awaiting_review' RETURNING`, else 409 | compass | A double click or the expiry cron can't resume the same thread twice |
| P10 | A malformed resume value means reject; an unknown interrupt kind is an error; the final status comes from the final graph state | compass | Keeps the audit trail consistent |
| P11 | After a decision the checkpoint thread is deleted; `triage_runs` is the audit trail | compass | Checkpoint tables don't grow forever |
| P12 | Reviews nobody answers for 7 days become `expired`, with no label written | compass (changed) | A timeout isn't a "no"; a fake label would poison the eval data |
| P13 | Each eval run writes a JSON artifact with its metrics, criteria version and a per-post failure list | compass | You fix the misses, not the average |
| P14 | Every failure found goes into `docs/KNOWN_ISSUES.md` with its evidence and a regression test before anything is changed | compass | Stops fixing one bug while breaking another |
| P15 | Architecture decision records in `docs/decisions/` | JobGraph | Decisions stay explained |
| P16 | Database code is tested against a real Postgres with `testcontainers` | JobGraph | Upsert and versioning SQL tested for real |

**Not taken:** compass's labeler shows the model's output while you label (that biases the labels), synthetic eval
records, LLM-as-judge scoring, and calibration of the screening score.

**What this project adds:** a cheap screening call (Jev) on every post, with the local LLM only reading the posts
Jev is unsure about (both prior projects send every post to an LLM); an arq worker with retries and idempotent job
ids; a Postgres checkpointer; Recruitee, Workable and Teamtailor fetchers; and an eval on hand labels with a frozen
tuning/held-out split and confidence intervals.
