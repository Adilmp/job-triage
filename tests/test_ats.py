from triage.ats import domain_stem_of, parse_ats_urls, slugs, trust_of


def test_stem_pops_gtlds():
    assert domain_stem_of("gigsa.solutions") == "gigsa"
    assert domain_stem_of("https://www.lorikeetcx.ai/about") == "lorikeetcx"
    assert domain_stem_of("synapse-analytics.io") == "synapseanalytics"
    assert domain_stem_of(None) == ""
    assert domain_stem_of("nodots") == ""


def test_slugs_put_domain_stem_first():
    assert slugs("Lorikeet", "lorikeetcx.ai")[:2] == ["lorikeetcx", "lorikeet"]
    assert slugs("Synapse Analytics", "synapse-analytics.io")[0] == "synapseanalytics"


def test_trust_blocks_stranger_boards():
    assert trust_of("cognitev", "cognitev.com") == "high"
    assert trust_of("lorikeet", "lorikeetcx.ai", "Lorikeet") == "high"  # full company name, prefix of the stem
    assert trust_of("digital", "digitalpetroleum.xyz", "Digital Petroleum") == "UNVERIFIED-slug-guess"
    assert trust_of("gro", "grosolutions.io", "GRO Solutions") == "UNVERIFIED-slug-guess"
    assert trust_of("gro", "grosolutions.io", "GRO Solutions", board_name="GRO Solutions") == "high"
    assert trust_of("x", "grosolutions.io", "GRO Solutions", board_name="Gross Domestic") == "UNVERIFIED-slug-guess"
    assert trust_of("pluralis-research", "pluralis.ai", "Pluralis Research") == "high"  # hyphenated full name
    assert trust_of("clarity", "onclarity.com", "Clarity", board_name="Clarity") == "UNVERIFIED-slug-guess"  # name alone is not proof
    assert trust_of("black", "black.ai", "black.ai", board_name="Acme Corp") == "UNVERIFIED-slug-guess"  # name vetoes a stem match
    assert trust_of("pluralis-research", "pluralis.ai", "Pluralis Research") == "high"  # hyphenated full name
    assert trust_of("clarity", "onclarity.com") == "UNVERIFIED-slug-guess"
    assert trust_of("ruyaai", "ruyatechai.com") == "UNVERIFIED-slug-guess"
    assert trust_of("ab", "abacus.ai") == "UNVERIFIED-slug-guess"      # too short for the prefix rule
    assert trust_of("solutions", "gigsa.solutions") == "UNVERIFIED-slug-guess"


def test_parse_ats_urls():
    assert parse_ats_urls("https://job-boards.eu.greenhouse.io/ruyaai") == [("greenhouse", "ruyaai")]
    assert parse_ats_urls("https://jobs.eu.lever.co/diabolocom") == [("lever", "diabolocom")]
    assert parse_ats_urls("https://apply.workable.com/lucidya/") == [("workable", "lucidya")]
    assert parse_ats_urls("https://cognitev.recruitee.com/") == [("recruitee", "cognitev")]
    assert parse_ats_urls("https://jobs.ashbyhq.com/eleven.labs") == [("ashby", "eleven.labs")]


def test_parse_ats_urls_in_html_skips_generic_hosts():
    html = '<script src="https://cdn.teamtailor.com/x.js"></script><a href="https://acme.teamtailor.com/jobs">'
    assert parse_ats_urls(html) == [("teamtailor", "acme")]
    assert parse_ats_urls("https://example.com/careers") == []
