from triage.fetchers.base import PermanentError, RawPost, get_json
from triage.normalize import strip_html


async def fetch(client, slug: str) -> list[RawPost]:
    data = await get_json(client, f"https://{slug}.recruitee.com/api/offers/")
    if not isinstance(data, dict) or "offers" not in data:
        raise PermanentError(f"recruitee:{slug} response has no 'offers'")
    posts = []
    for o in data["offers"]:
        if o.get("status", "published") != "published":
            continue
        # The role description and the requirements are two separate HTML fields.
        text = "\n\n".join(t for t in (strip_html(o.get("description")), strip_html(o.get("requirements"))) if t)
        posts.append(RawPost(
            provider="recruitee",
            external_id=str(o["id"]),
            url=o.get("careers_url", ""),
            title=o.get("title", ""),
            location=o.get("location") or ", ".join(x for x in (o.get("city"), o.get("country")) if x) or None,
            description_text=text,
            raw=o,
        ))
    return posts
