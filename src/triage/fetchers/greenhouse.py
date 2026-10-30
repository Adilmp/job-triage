from triage.fetchers.base import PermanentError, RawPost, get_json
from triage.normalize import strip_html


async def fetch(client, slug: str) -> list[RawPost]:
    # content=true is what puts the description in the list response; Greenhouse returns it HTML-escaped (P1).
    data = await get_json(client, f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs", {"content": "true"})
    if not isinstance(data, dict) or "jobs" not in data:
        raise PermanentError(f"greenhouse:{slug} response has no 'jobs'")
    return [
        RawPost(
            provider="greenhouse",
            external_id=str(j["id"]),
            url=j.get("absolute_url", ""),
            title=j.get("title", ""),
            location=(j.get("location") or {}).get("name") or None,
            description_text=strip_html(j.get("content")),
            raw=j,
        )
        for j in data["jobs"]
    ]
