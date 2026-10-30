from triage.fetchers.base import PermanentError, RawPost, get_json
from triage.normalize import strip_html


def _location(j: dict) -> str | None:
    return ", ".join(dict.fromkeys(x for x in (j.get("city"), j.get("state"), j.get("country")) if x)) or None


async def fetch(client, slug: str) -> list[RawPost]:
    data = await get_json(client, f"https://apply.workable.com/api/v1/widget/accounts/{slug}", {"details": "true"})
    if not isinstance(data, dict) or "jobs" not in data:
        raise PermanentError(f"workable:{slug} response has no 'jobs'")
    return [
        RawPost(
            provider="workable",
            external_id=j["shortcode"],
            url=j.get("url") or j.get("shortlink", ""),
            title=j.get("title", ""),
            location=_location(j),
            description_text=strip_html(j.get("description")),
            raw=j,
        )
        for j in data["jobs"]
    ]
