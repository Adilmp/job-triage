"""Teamtailor has no JSON API; every career site serves an RSS feed. Best-effort: a company that turns the
feed off, or a page that isn't RSS, is a PermanentError. There is deliberately no HTML-scraping fallback:
it yields titles and locations but no descriptions, and posts without descriptions are dropped (P2).

The slug is the teamtailor.com subdomain, or a full host when the career site is on the company's own
domain (`careers.acme.co`), which is how probe.py stores those."""
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from triage.fetchers.base import PermanentError, RawPost, get_text
from triage.normalize import strip_html


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _item_raw(item) -> dict:
    """Child elements by tag name (so derive() can read `remoteStatus`), plus the list of locations."""
    raw = {}
    for child in item:
        name = _local(child.tag)
        if name == "locations":
            raw["locations"] = [
                {_local(f.tag): (f.text or "").strip() for f in loc} for loc in child
            ]
        elif name not in raw:
            raw[name] = (child.text or "").strip()
    return raw


def _location(raw: dict) -> str | None:
    places = [", ".join(dict.fromkeys(x for x in (l.get("city") or l.get("name"), l.get("country")) if x))
              for l in raw.get("locations", [])]
    return " | ".join(dict.fromkeys(p for p in places if p)) or None


async def fetch(client, slug: str) -> list[RawPost]:
    host = slug if "." in slug else f"{slug}.teamtailor.com"
    text = await get_text(client, f"https://{host}/jobs.rss")
    try:
        root = ElementTree.fromstring(text.encode())
    except (ElementTree.ParseError, DefusedXmlException) as e:
        raise PermanentError(f"teamtailor:{slug} did not return a valid RSS feed ({type(e).__name__})") from e
    if _local(root.tag) != "rss":
        raise PermanentError(f"teamtailor:{slug} did not return an RSS feed")
    posts = []
    for item in root.iter("item"):
        raw = _item_raw(item)
        posts.append(RawPost(
            provider="teamtailor",
            external_id=raw.get("guid") or raw.get("link", ""),
            url=raw.get("link", ""),
            title=raw.get("title", ""),
            location=_location(raw),
            description_text=strip_html(raw.get("description")),
            raw=raw,
        ))
    return posts
