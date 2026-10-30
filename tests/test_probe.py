import httpx
import respx

from triage import probe


@respx.mock
async def test_long_retry_after_trips_breaker_and_stops_requests():
    route = respx.get(url__regex=r"https://apply\.workable\.com/.*").mock(
        return_value=httpx.Response(429, headers={"retry-after": "86000"}))
    probe._blocked_until.clear()
    async with httpx.AsyncClient() as client:
        assert await probe.board_info(client, "workable", "acme") is None
        assert await probe.board_info(client, "workable", "other") is None
    assert route.call_count == 1                      # second call never left the process
    assert probe.rate_limited_sites() == ["workable.com"]
    probe._blocked_until.clear()


@respx.mock
async def test_404_means_no_board_and_is_not_retried():
    route = respx.get(url__regex=r"https://api\.ashbyhq\.com/.*").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        assert await probe.board_info(client, "ashby", "nope") is None
    assert route.call_count == 1


@respx.mock
async def test_board_info_reads_counts_and_names():
    respx.get("https://apply.workable.com/api/v1/widget/accounts/acme?details=true").mock(
        return_value=httpx.Response(200, json={"name": "Acme", "jobs": [{}, {}]}))
    respx.get("https://acme.recruitee.com/api/offers/").mock(
        return_value=httpx.Response(200, json={"offers": [{"company_name": "Acme Ltd"}]}))
    probe._blocked_until.clear()
    async with httpx.AsyncClient() as client:
        assert await probe.board_info(client, "workable", "acme") == (2, "Acme")
        assert await probe.board_info(client, "recruitee", "acme") == (1, "Acme Ltd")


@respx.mock
async def test_teamtailor_feed_title_is_the_board_name():
    rss = '<rss><channel><title>Eventus</title><item><title>Job</title></item></channel></rss>'
    respx.get("https://black.teamtailor.com/jobs.rss").mock(return_value=httpx.Response(200, text=rss))
    async with httpx.AsyncClient() as client:
        assert await probe.board_info(client, "teamtailor", "black") == (1, "Eventus")


async def test_probe_does_not_claim_a_board_another_company_owns(session):
    from triage.models import Company
    a = Company(name="Anthropic", region="BACKFILL", board="greenhouse", slug="anthropic", verified=True)
    b = Company(name="Anthropic AI", region="REMOTE")
    c = Company(name="Other", region="REMOTE")
    session.add_all([a, b, c])
    await session.commit()
    await probe.save_result(session, b.id, probe.Candidate("greenhouse", "anthropic", "high", 5), None)
    await probe.save_result(session, c.id, probe.Candidate("ashby", "other", "high", 2), None)
    await session.commit()
    await session.refresh(b), await session.refresh(c)
    assert b.board is None and "already belongs to Anthropic" in b.last_error
    assert (c.board, c.slug, c.verified) == ("ashby", "other", True)


@respx.mock
async def test_workable_request_budget_stops_the_probe_before_a_ban(monkeypatch):
    route = respx.get(url__regex=r"https://apply\.workable\.com/.*").mock(return_value=httpx.Response(404))
    probe._blocked_until.clear(), probe._spent.clear()
    monkeypatch.setitem(probe._BUDGET, "workable.com", 3)
    async with httpx.AsyncClient() as client:
        for i in range(6):
            assert await probe.board_info(client, "workable", f"slug{i}") is None
    assert route.call_count == 3 and probe.rate_limited_sites() == ["workable.com"]
    probe._blocked_until.clear(), probe._spent.clear()
