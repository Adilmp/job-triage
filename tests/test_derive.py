import pytest

from triage.derive import derive


def r(provider, raw, location=None):
    return derive(provider, raw, location).remote_eligibility


# --- Ashby: workplaceType wins; isRemote is the fallback; conflicts are 'unclear' (P4, P5) ---
@pytest.mark.parametrize("raw,loc,want", [
    ({"isRemote": True, "workplaceType": "Remote"}, "Remote", "eligible"),
    ({"isRemote": True, "workplaceType": "Remote"}, "US - Remote", "region_locked"),
    ({"isRemote": False, "workplaceType": "OnSite"}, "Sydney, Australia HQ", "onsite"),
    ({"isRemote": False, "workplaceType": "Hybrid"}, "London", "hybrid"),
    ({"isRemote": True}, "Anywhere", "eligible"),                       # no workplaceType: isRemote decides
    ({"isRemote": False}, "Sydney", "unclear"),                         # False alone doesn't prove onsite
    ({"isRemote": True, "workplaceType": "Hybrid"}, "San Francisco", "unclear"),   # real Vapi payload: typed fields disagree
    ({}, "Remote", "unclear"),                                          # no typed field: don't guess from text
])
def test_ashby(raw, loc, want):
    assert r("ashby", raw, loc) == want


# --- Lever ---
@pytest.mark.parametrize("raw,loc,want", [
    ({"workplaceType": "remote"}, "North America", "region_locked"),
    ({"workplaceType": "remote"}, "Remote - Worldwide", "eligible"),
    ({"workplaceType": "hybrid"}, "Singapore", "hybrid"),
    ({"workplaceType": "onsite"}, "Paris", "onsite"),
    ({"workplaceType": "unspecified"}, "Remote", "unclear"),
    ({}, None, "unclear"),
])
def test_lever(raw, loc, want):
    assert r("lever", raw, loc) == want


# --- Recruitee: three booleans ---
@pytest.mark.parametrize("raw,want", [
    ({"remote": True, "on_site": False, "hybrid": False}, "region_locked"),   # location names Cairo below
    ({"remote": False, "on_site": True, "hybrid": False}, "onsite"),
    ({"remote": False, "on_site": False, "hybrid": True}, "hybrid"),
    ({"remote": False, "on_site": False, "hybrid": False}, "unclear"),
    ({"remote": True, "on_site": True, "hybrid": False}, "unclear"),          # contradictory
])
def test_recruitee(raw, want):
    assert r("recruitee", raw, "Cairo, Egypt") == want


def test_teamtailor_remote_status():
    assert r("teamtailor", {"remoteStatus": "fully"}, "Cairo, Lahore, Karachi") == "region_locked"
    assert r("teamtailor", {"remoteStatus": "fully"}, "Remote") == "eligible"
    assert r("teamtailor", {"remoteStatus": "hybrid"}, "Dubai") == "hybrid"
    assert r("teamtailor", {"remoteStatus": "onsite"}, "Dubai") == "onsite"
    assert r("teamtailor", {"remoteStatus": "none"}, "Dubai") == "unclear"
    assert r("teamtailor", {"remoteStatus": "temporary"}, "Dubai") == "unclear"


def test_workable_telecommuting_flag():
    assert r("workable", {"telecommuting": True}, "Remote") == "eligible"
    assert r("workable", {"telecommuting": False}, "Dubai") == "unclear"      # False = not remote, onsite vs hybrid unknown
    assert r("workable", {}, "Dubai") == "unclear"


# --- Greenhouse: custom 'Location Type' metadata when present, else cautious location text ---
def test_greenhouse_metadata_and_text_fallback():
    meta = lambda v: {"metadata": [{"name": "Location Type", "value": v}]}
    assert r("greenhouse", meta("On-Site"), "Singapore") == "onsite"
    assert r("greenhouse", meta("Hybrid (Travel-Required)"), "New York City, NY") == "hybrid"
    assert r("greenhouse", meta("Remote"), "London, UK") == "region_locked"
    assert r("greenhouse", {"metadata": [{"name": "Location Type", "value": None}]}, "Remote") == "eligible"   # None -> text
    assert r("greenhouse", {"metadata": [{"name": "Salary band", "value": "L5"}]}, "Singapore") == "unclear"   # unrelated field
    assert r("greenhouse", {}, "Remote - Anywhere") == "eligible"
    assert r("greenhouse", {}, "Remote - US") == "region_locked"
    assert r("greenhouse", {}, "Hybrid, Berlin") == "hybrid"
    assert r("greenhouse", {}, "Remote or Hybrid, Berlin") == "unclear"      # both signals: don't pick one
    assert r("greenhouse", {}, "San Francisco, CA") == "unclear"             # a city name proves nothing
    assert r("greenhouse", {}, None) == "unclear"


def test_remote_global_wording():
    for loc in ("Remote", "Remote (Worldwide)", "Anywhere", "Global - Remote", "Work from anywhere", "Remote, Global"):
        assert r("lever", {"workplaceType": "remote"}, loc) == "eligible", loc
    for loc in ("Remote - EMEA", "Remote (Europe)", "Remote, Pakistan", "UK - Remote"):
        assert r("lever", {"workplaceType": "remote"}, loc) == "region_locked", loc
    assert r("lever", {"workplaceType": "remote"}, None) == "eligible"       # typed remote, no place named


def test_unknown_provider_is_unclear():
    assert r("somethingnew", {"remote": True}, "Remote") == "unclear"


# --- Compensation (P3): only when the ATS says to display it ---
def ashby_comp(display, components):
    return {"shouldDisplayCompensationOnJobPostings": display, "compensation": {"summaryComponents": components}}


SALARY = {"compensationType": "Salary", "interval": "1 YEAR", "currencyCode": "USD", "minValue": 150000, "maxValue": 175000}
EQUITY = {"compensationType": "EquityPercentage", "interval": "NONE", "currencyCode": None, "minValue": None, "maxValue": None}


def test_ashby_comp_shown():
    d = derive("ashby", ashby_comp(True, [EQUITY, SALARY]), "Remote")
    assert (d.comp_min, d.comp_max, d.comp_currency) == (150000, 175000, "USD")


def test_ashby_comp_hidden_or_missing_is_ignored():
    assert derive("ashby", ashby_comp(False, [SALARY]), "x").comp_min is None
    assert derive("ashby", {"compensation": {"summaryComponents": [SALARY]}}, "x").comp_min is None
    assert derive("ashby", ashby_comp(True, []), "x").comp_min is None
    assert derive("ashby", ashby_comp(True, [EQUITY]), "x").comp_min is None
    # older payload spelling of the flag
    old = {"shouldDisplayCompensationOnJobBoard": True, "compensation": {"summaryComponents": [SALARY]}}
    assert derive("ashby", old, "x").comp_min == 150000


def test_ashby_comp_monthly_is_annualized_and_hourly_is_skipped():
    monthly = {**SALARY, "interval": "1 MONTH", "minValue": 5000, "maxValue": 6000}
    d = derive("ashby", ashby_comp(True, [monthly]), "x")
    assert (d.comp_min, d.comp_max) == (60000, 72000)
    hourly = {**SALARY, "interval": "1 HOUR", "minValue": 40, "maxValue": 60}
    assert derive("ashby", ashby_comp(True, [hourly]), "x").comp_min is None


def test_lever_and_recruitee_salary():
    lever = {"salaryRange": {"currency": "USD", "interval": "per-year-salary", "min": 120000, "max": 160000}}
    d = derive("lever", lever, "x")
    assert (d.comp_min, d.comp_max, d.comp_currency) == (120000, 160000, "USD")
    assert derive("lever", {"salaryRange": {"currency": "USD", "interval": "per-hour-wage", "min": 50, "max": 70}}, "x").comp_min is None
    rec = {"salary": {"min": 3000, "max": 4000, "period": "month", "currency": "EUR"}}
    d = derive("recruitee", rec, "x")
    assert (d.comp_min, d.comp_max, d.comp_currency) == (36000, 48000, "EUR")
    assert derive("recruitee", {"salary": {"min": None, "max": None, "period": None, "currency": None}}, "x").comp_min is None
