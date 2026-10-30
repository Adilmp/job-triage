# Known issues

Every failure found gets an entry with its evidence, and a regression test, before any wording or rule changes (P14 in [DESIGN.md](DESIGN.md)).

## 1. "Member of Technical Staff" dropped as a staff-level role (prefilter) - FIXED

- **Found:** 2026-10-30, `run_eval.py tune` on the tuning split. A gold positive, "Member of Technical Staff (AI ...)",
  never reached Jev: the prefilter's seniority rule matched the word "staff".
- **Why it matters:** "Member of Technical Staff" is the ordinary individual-contributor title at many AI labs
  (OpenAI, Anthropic, Inflection, ...). Dropping it silently loses exactly the roles this project is for.
- **Fix:** the seniority regex skips "staff" when it follows "technical ". Regression case in
  `tests/test_prefilter.py` (`test_other_titles_pass`).

## 2. Jev's raw scores are compressed near zero (screening) - OPEN

- **Found:** 2026-10-30, `tune` on the tuning split (108 posts past the prefilter, 29 positive).
- **Evidence:** AUC 0.77, so Jev ranks fits above non-fits more often than not, but the scores barely spread out.
  Positives' median score is 0.04, negatives' median is 0.02, and 22 of 29 positives score at or below 0.07.
  With the default LO=0.2, 24 of 29 positives (83%) would be auto-rejected.
- **Consequence:** no LO both keeps the false-reject rate near zero and spares the deep read much work.
  LO=0.005 misses none but sends 105 of 108 posts to the deep read; LO=0.02 sends 41 but misses 7 positives.
- **Not yet decided:** keep the frozen wording and live with a low LO (the deep read does most of the work), or
  treat this as a finding about the wording and change it, which would mean the held-out run must say so.
