from triage.seed import AU_CSV, GULF_CSV, read_backfill, read_companies


def test_au_and_gulf_csvs_load():
    au = read_companies(AU_CSV, "AU", "company", "website")
    gulf = read_companies(GULF_CSV, "GULF", "Company", "Website")
    assert len(au) >= 60 and len(gulf) >= 45
    assert {"name", "website", "region"} == set(au[0])
    assert all(r["name"] for r in au + gulf)


def test_backfill_only_ats_boards():
    rows = read_backfill()
    assert len(rows) > 50
    assert {r["board"] for r in rows} <= {"greenhouse", "lever", "ashby"}
    assert all(r["slug"] and r["region"] == "BACKFILL" for r in rows)
    eleven = next(r for r in rows if r["name"] == "ElevenLabs")
    assert (eleven["board"], eleven["slug"]) == ("ashby", "elevenlabs")


def test_placeholder_websites_become_none():
    from triage.seed import clean_website
    assert clean_website("unknown (search: Kimia chemical intelligence platform)") is None
    assert clean_website("lorikeetcx.ai") == "lorikeetcx.ai"
    assert clean_website("https://www.acme.co/about") == "https://www.acme.co/about"
    assert clean_website("") is None
