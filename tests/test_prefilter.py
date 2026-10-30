import pytest

from triage.services.prefilter import prefilter


def post(title="ML Engineer", location="Remote", remote="eligible"):
    return {"title": title, "location": location, "remote_eligibility": remote}


@pytest.mark.parametrize("title", [
    "Staff Software Engineer", "Principal Engineer", "Head of AI", "Director of Engineering", "VP Engineering",
    "Vice President, Data", "Chief Scientist", "Engineering Manager", "Software Engineer IV",
    "Staff/Principal Data Scientist (Optimisation)", "Distinguished Engineer",
])
def test_clearly_too_high_titles_are_dropped(title):
    r = prefilter(post(title))
    assert r.drop and "seniority" in r.reason


@pytest.mark.parametrize("title", [
    "ML Engineer", "Junior Backend Engineer", "AI Engineer (LLM)", "Software Engineer, Applied AI",
    "Associate Data Scientist", "Research Engineer - Post-Training", "Software Engineer II",
    "Leadership Development Intern",           # 'lead' inside a longer word must not match
    "Misleading Title Engineer",
    # Companies sometimes hire juniors into senior-titled posts, so these reach Jev, which reads the years asked.
    "Senior ML Engineer", "Sr. Backend Engineer", "Lead Data Scientist", "Software Engineer III",
    "Senior Forward Deployed Engineer",
    "Member of Technical Staff (AI Engineering)",      # KNOWN_ISSUES #1: an IC title at AI labs, found on the tuning split
])
def test_other_titles_pass(title):
    assert not prefilter(post(title)).drop


@pytest.mark.parametrize("remote,location", [
    ("onsite", "San Francisco, CA"), ("hybrid", "London, UK"), ("onsite", "Dubai, UAE"), ("hybrid", None),
])
def test_onsite_and_hybrid_outside_targets_are_dropped(remote, location):
    r = prefilter(post(location=location, remote=remote))
    assert r.drop and remote in r.reason


@pytest.mark.parametrize("remote,location", [
    ("onsite", "Karachi, Pakistan"), ("hybrid", "Lahore"), ("onsite", "Remote | Islamabad"),
])
def test_onsite_and_hybrid_in_targets_pass(remote, location):
    assert not prefilter(post(location=location, remote=remote)).drop


@pytest.mark.parametrize("remote", ["eligible", "region_locked", "unclear"])
def test_remote_and_unclear_always_pass_the_location_rule(remote):
    # P5: reject only onsite/hybrid outside targets. 'unclear' goes through, and so does region_locked.
    assert not prefilter(post(location="Berlin", remote=remote)).drop


def test_targets_come_from_settings(monkeypatch):
    from triage.config import settings
    monkeypatch.setattr(settings, "TARGET_LOCATIONS", "dubai, riyadh")
    assert not prefilter(post(location="Dubai, UAE", remote="onsite")).drop
    assert prefilter(post(location="Karachi", remote="onsite")).drop


def test_reason_is_empty_when_kept():
    r = prefilter(post())
    assert (r.drop, r.reason) == (False, "")


@pytest.mark.parametrize("text", [
    "This role is open to remote candidates.", "You can work remotely from anywhere in EMEA.", "Hybrid or remote.",
    "We are remote-first.", "Remote options available for the right person.",
])
def test_onsite_post_that_offers_remote_in_its_text_goes_through(text):
    p = {**post(location="London", remote="onsite"), "description_text": f"Join us. {text} Apply now."}
    assert not prefilter(p).drop


@pytest.mark.parametrize("text", [
    "Our remote work policy is described in the handbook.",      # bare 'remote' is boilerplate, not an offer
    "Three days a week in our London office.", "", None,
])
def test_onsite_post_without_a_remote_offer_is_still_dropped(text):
    p = {**post(location="London", remote="onsite"), "description_text": text}
    assert prefilter(p).drop
