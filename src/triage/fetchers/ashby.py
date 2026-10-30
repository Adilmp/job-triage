from triage.fetchers.base import PermanentError, RawPost, get_json
from triage.normalize import strip_html, tidy_text


def _location(j: dict) -> str | None:
    # Payload versions spell the primary location `location` or `locationName` (P3). Secondary locations
    # ("US - Remote" next to a Sydney HQ) are appended so location filters see every place the role is open.
    places = [j.get("location") or j.get("locationName")] + [s.get("location") for s in j.get("secondaryLocations") or []]
    return " | ".join(dict.fromkeys(p.strip() for p in places if p and p.strip())) or None


def _description(j: dict) -> str:
    return tidy_text(j.get("descriptionPlain")) or strip_html(j.get("descriptionHtml") or j.get("description"))


async def fetch(client, slug: str) -> list[RawPost]:
    data = await get_json(client, f"https://api.ashbyhq.com/posting-api/job-board/{slug}", {"includeCompensation": "true"})
    if not isinstance(data, dict) or data.get("jobs") is None:
        raise PermanentError(f"ashby:{slug} response has no 'jobs'")
    return [
        RawPost(
            provider="ashby",
            external_id=j["id"],
            url=j.get("jobUrl", ""),
            title=j.get("title", ""),
            location=_location(j),
            description_text=_description(j),
            raw=j,
        )
        for j in data["jobs"]
        if j.get("isListed", True)
    ]
