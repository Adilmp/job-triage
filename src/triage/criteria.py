"""criteria_version: a hash of everything that decides a verdict (P7).

Stored on every triage run. When the Jev wording, the prefilter rules or the thresholds change, the version
changes, and POST /retriage?stale=true re-runs only the posts whose last run used an older one.
"""
import hashlib
import json

from triage.config import settings
from triage.services import deep_read, jev, prefilter


def _ingredients() -> dict:
    return {
        "jev_questions": jev.QUESTIONS,
        "prefilter_title": prefilter.SENIOR_TITLE.pattern,
        "prefilter_targets": prefilter.targets(),
        "hi": settings.HI,
        "lo": settings.LO,
        "min_conf": settings.DEEP_READ_MIN_CONFIDENCE,
        "deep_read_prompt": deep_read.SYSTEM_PROMPT,
    }


def criteria_version() -> str:
    blob = json.dumps(_ingredients(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
