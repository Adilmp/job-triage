"""Remote eligibility and compensation read from the ATS's typed fields, before any model sees the post (P4).

`raw` is the provider's native payload for one job. For Teamtailor (RSS) the fetcher stores each item's child
elements under their tag names, e.g. {"remoteStatus": "fully"}. `location` is the display string the fetcher
extracted. Anything the typed fields can't settle is 'unclear', never a guess (P5): a wrong eligible/onsite
costs more than "don't know", and 'unclear' posts still go on to screening.

remote_eligibility: eligible (remote, no place named) | region_locked (remote, but a place is named)
                    | hybrid | onsite | unclear
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Derived:
    remote_eligibility: str
    comp_min: float | None = None
    comp_max: float | None = None
    comp_currency: str | None = None


_GLOBAL = re.compile(r"\b(anywhere|worldwide|global(ly)?|any location|world)\b", re.I)
_REMOTE_FILLER = re.compile(
    r"\b(remote(ly)?|work from home|wfh|work from anywhere|telecommut\w*|fully|friendly|only|based|distributed|virtual)\b", re.I)
_TEXT_REMOTE = re.compile(r"\b(remote(ly)?|work from home|wfh|work from anywhere)\b", re.I)
_TEXT_HYBRID = re.compile(r"\bhybrid\b", re.I)


def _kind(value) -> str | None:
    """Map an ATS workplace label (Ashby 'OnSite', Lever 'remote', Greenhouse 'Hybrid (Travel-Required)',
    Teamtailor 'fully') to remote|hybrid|onsite, or None for 'unspecified', 'none', 'temporary', etc."""
    if not isinstance(value, str):
        return None
    t = re.sub(r"[^a-z]", "", value.lower())
    if t.startswith("hybrid"):
        return "hybrid"
    if t.startswith("remote") or t in ("fully", "fullyremote"):
        return "remote"
    if t.startswith("onsite") or t in ("office", "inoffice"):
        return "onsite"
    return None


def _remote(location: str | None) -> str:
    """Typed as remote: open to anyone if no place is named (or the wording is global), else tied to a region."""
    if not location or _GLOBAL.search(location):
        return "eligible"
    leftover = re.sub(r"[\W_]+", "", _REMOTE_FILLER.sub(" ", location))
    return "region_locked" if leftover else "eligible"


def _resolve(kind: str | None, location: str | None) -> str:
    return {"remote": _remote(location), "hybrid": "hybrid", "onsite": "onsite"}.get(kind, "unclear")


def _from_text(location: str | None) -> str:
    """Last resort for boards with no typed field (Greenhouse, Workable). Only explicit words count: a city name
    proves nothing, and a location naming both remote and hybrid is ambiguous."""
    text = location or ""
    remote, hybrid = bool(_TEXT_REMOTE.search(text)), bool(_TEXT_HYBRID.search(text))
    if remote == hybrid:
        return "unclear"
    return _remote(location) if remote else "hybrid"


def _ashby(raw: dict, location: str | None) -> str:
    kind, is_remote = _kind(raw.get("workplaceType")), raw.get("isRemote")
    if kind is None:
        return _resolve("remote" if is_remote is True else None, location)
    if (is_remote is True) != (kind == "remote") and is_remote is not None:
        return "unclear"        # e.g. isRemote=true with workplaceType=Hybrid: the typed fields disagree
    return _resolve(kind, location)


def _recruitee(raw: dict, location: str | None) -> str:
    on = [k for k in ("remote", "hybrid", "on_site") if raw.get(k)]
    return _resolve(_kind(on[0].replace("_", "")), location) if len(on) == 1 else "unclear"


def _greenhouse(raw: dict, location: str | None) -> str:
    # Many boards add a custom "Location Type" field; its name varies, its absence is common.
    for m in raw.get("metadata") or []:
        if re.search(r"location type|work(place)? (type|model)|remote", str(m.get("name", "")), re.I):
            kind = _kind(m.get("value"))
            if kind:
                return _resolve(kind, location)
    return _from_text(location)


def _remote_eligibility(provider: str, raw: dict, location: str | None) -> str:
    if provider == "ashby":
        return _ashby(raw, location)
    if provider == "lever":
        return _resolve(_kind(raw.get("workplaceType")), location)
    if provider == "recruitee":
        return _recruitee(raw, location)
    if provider == "teamtailor":
        return _resolve(_kind(raw.get("remoteStatus")), location)
    if provider == "workable":      # telecommuting=false only says "not remote": onsite vs hybrid is unknown
        return _resolve("remote" if raw.get("telecommuting") is True else None, location)
    if provider == "greenhouse":
        return _greenhouse(raw, location)
    return "unclear"


def _annual(interval, lo, hi) -> tuple[float, float] | None:
    """Per-year figures from a yearly or monthly pay interval. Hourly/weekly/one-off are skipped, not guessed."""
    text = str(interval or "").lower()
    factor = 1 if ("year" in text or "annual" in text) else 12 if "month" in text else None
    if factor is None or (lo is None and hi is None):
        return None
    return (lo * factor if lo is not None else None, hi * factor if hi is not None else None)


def _comp(provider: str, raw: dict) -> tuple[float | None, float | None, str | None]:
    none = (None, None, None)
    if provider == "ashby":
        # Only when the board says to display it (P3). The flag's spelling differs between payload versions.
        if not (raw.get("shouldDisplayCompensationOnJobPostings") or raw.get("shouldDisplayCompensationOnJobBoard")):
            return none
        for c in (raw.get("compensation") or {}).get("summaryComponents") or []:
            if c.get("compensationType") == "Salary" and (a := _annual(c.get("interval"), c.get("minValue"), c.get("maxValue"))):
                return (*a, c.get("currencyCode"))
    elif provider == "lever":
        s = raw.get("salaryRange") or {}
        if a := _annual(s.get("interval"), s.get("min"), s.get("max")):
            return (*a, s.get("currency"))
    elif provider == "recruitee":
        s = raw.get("salary") or {}
        if a := _annual(s.get("period"), s.get("min"), s.get("max")):
            return (*a, s.get("currency"))
    return none


def derive(provider: str, raw: dict, location: str | None) -> Derived:
    return Derived(_remote_eligibility(provider, raw, location), *_comp(provider, raw))
