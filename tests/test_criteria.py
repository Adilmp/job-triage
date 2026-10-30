from triage import criteria
from triage.config import settings
from triage.services import jev, prefilter


def test_version_is_stable_and_short():
    v = criteria.criteria_version()
    assert v == criteria.criteria_version() and len(v) == 16


def test_version_changes_with_each_ingredient(monkeypatch):
    base = criteria.criteria_version()
    seen = {base}

    def changed():
        v = criteria.criteria_version()
        assert v not in seen
        seen.add(v)

    monkeypatch.setitem(jev.QUESTIONS["fit"], "instructions", "Different wording?")
    changed()
    monkeypatch.setattr(settings, "TARGET_LOCATIONS", "dubai")
    changed()
    monkeypatch.setattr(settings, "HI", 0.9)
    changed()
    monkeypatch.setattr(settings, "LO", 0.05)
    changed()
    monkeypatch.setattr(prefilter, "SENIOR_TITLE", __import__("re").compile(r"\bgrand\b"))
    changed()


def test_version_changes_with_the_deep_read_prompt(monkeypatch):
    from triage.services import deep_read
    base = criteria.criteria_version()
    monkeypatch.setattr(deep_read, "SYSTEM_PROMPT", deep_read.SYSTEM_PROMPT + " Be stricter.")
    assert criteria.criteria_version() != base


def test_version_changes_with_the_deep_read_confidence_line(monkeypatch):
    base = criteria.criteria_version()
    monkeypatch.setattr(settings, "DEEP_READ_MIN_CONFIDENCE", 0.5)
    assert criteria.criteria_version() != base
