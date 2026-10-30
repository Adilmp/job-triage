from triage.fetchers.base import PermanentError, RawPost, get_json
from triage.normalize import strip_html, tidy_text


def _description(j: dict) -> str:
    # descriptionPlain is the body; the requirement sections and closing paragraph come as separate fields.
    parts = [j.get("openingPlain") or "", j.get("descriptionPlain") or ""]
    for section in j.get("lists") or []:
        parts.append(f"{section.get('text', '')}\n{strip_html(section.get('content'))}")
    parts.append(j.get("additionalPlain") or "")
    return tidy_text("\n\n".join(p.strip() for p in parts if p and p.strip()))


def _location(j: dict) -> str | None:
    cats = j.get("categories") or {}
    return " | ".join(cats.get("allLocations") or []) or cats.get("location") or None


async def fetch(client, slug: str) -> list[RawPost]:
    data = await get_json(client, f"https://api.lever.co/v0/postings/{slug}", {"mode": "json"})
    if not isinstance(data, list):
        # An unknown board is a 200 with {"ok": false, "error": "Document not found"}.
        raise PermanentError(f"lever:{slug} {data.get('error', 'unexpected response') if isinstance(data, dict) else 'unexpected response'}")
    return [
        RawPost(
            provider="lever",
            external_id=j["id"],
            url=j.get("hostedUrl", ""),
            title=j.get("text", ""),
            location=_location(j),
            description_text=_description(j),
            raw=j,
        )
        for j in data
    ]
