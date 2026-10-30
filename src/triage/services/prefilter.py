"""Deterministic hard rules, run before any paid or slow model sees a post (the graph's first node).

Only two rules, both about things no screening model should be second-guessing:
  1. clearly-too-high titles: staff, principal, director, VP, head of, chief (the user is early-career)
  2. on-site or hybrid outside TARGET_LOCATIONS (the user is Pakistan-based and won't relocate), unless the post's
     own text offers remote work: a typed "on-site" flag and "remote possible" in the description both happen
Per P5 only 'onsite'/'hybrid' are rejected on location: 'unclear' and 'region_locked' go through.

`post` is a plain dict (the graph state holds plain dicts): title, location, remote_eligibility, description_text.
"""
import re
from dataclasses import dataclass

from triage.config import settings

# Only titles that are clearly far above early-career. "Senior"/"Lead"/"III" are NOT here: companies sometimes hire
# juniors into senior-titled posts, so those reach Jev, which reads the years the post actually asks for.
# "Member of Technical Staff" is an ordinary individual-contributor title at AI labs, not a staff-level role
# (docs/KNOWN_ISSUES.md #1), hence the lookbehind.
SENIOR_TITLE = re.compile(
    r"\b((?<!technical )staff|principal|head of|director|vp|vice president|chief|distinguished|engineering manager|iv)\b",
    re.I)


# On-site/hybrid posts that say, in the text, that remote is possible. Deliberately narrow: plain "remote" appears
# in about half of all on-site posts as boilerplate, so only phrases that offer remote work count.
REMOTE_HINT = re.compile(
    r"open to remote|remote(ly)? (possible|option|available|friendly|eligible)|can work remotely|work remotely|"
    r"fully remote|remote[- ]first|remote or\b|or remote\b|work from anywhere", re.I)


@dataclass(frozen=True)
class PrefilterResult:
    drop: bool
    reason: str = ""


def targets() -> list[str]:
    return [t.strip().lower() for t in settings.TARGET_LOCATIONS.split(",") if t.strip()]


def prefilter(post: dict) -> PrefilterResult:
    if m := SENIOR_TITLE.search(post.get("title") or ""):
        return PrefilterResult(True, f"seniority: title has '{m.group(0)}'")
    remote = post.get("remote_eligibility")
    if remote in ("onsite", "hybrid"):
        location = (post.get("location") or "").lower()
        if not any(t in location for t in targets()) and not REMOTE_HINT.search(post.get("description_text") or ""):
            return PrefilterResult(True, f"{remote} outside targets: {post.get('location') or 'no location given'}")
    return PrefilterResult(False)
