"""Pure helpers for finding a company's ATS board: slug guesses, stranger-board guard, URL parsing.

Ported from gulf-ai-startups-research/probe_ats.py (slugs, domain_stem_of, GTLDS, trust check).
"""
import re

GTLDS = {
    "solutions", "technology", "systems", "agency", "consulting", "digital",
    "online", "store", "cloud", "global", "world", "group", "network", "media",
    "studio", "space", "works", "today", "tools", "zone", "site", "live", "life",
    "team", "company", "center", "academy", "capital", "ventures", "software",
}

# Slugs these patterns capture that are never a company's board (CDN hosts, embeds, marketing pages).
_GENERIC_SLUGS = {"www", "api", "app", "apps", "static", "assets", "cdn", "embed", "careers", "jobs", "help", "support"}

# host/path -> board. Group 1 is the slug. Used both on a CSV careers_url and on a company's own careers HTML.
_ATS_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("greenhouse", re.compile(r"(?:boards|job-boards)(?:\.eu)?\.greenhouse\.io/(?:embed/job_board\?for=)?([a-z0-9_-]+)", re.I)),
    ("lever", re.compile(r"jobs\.(?:eu\.)?lever\.co/([a-z0-9_-]+)", re.I)),
    ("ashby", re.compile(r"(?:jobs\.ashbyhq\.com|api\.ashbyhq\.com/posting-api/job-board)/([a-z0-9_.-]+)", re.I)),
    ("workable", re.compile(r"apply\.workable\.com/(?:api/v\d/widget/accounts/)?([a-z0-9_-]+)", re.I)),
    ("recruitee", re.compile(r"([a-z0-9-]+)\.recruitee\.com", re.I)),
    ("teamtailor", re.compile(r"([a-z0-9-]+)\.teamtailor\.com", re.I)),
)


def _host(website: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", website.strip().lower()).split("/")[0]


def domain_stem_of(website: str | None) -> str:
    """Registrable name. Pops multi-char gTLDs too, or gigsa.solutions yields 'solutions',
    which then matches a stranger's job board."""
    if not isinstance(website, str) or "." not in website:
        return ""
    parts = [p for p in _host(website).split(".") if p]
    while len(parts) > 1 and (len(parts[-1]) <= 4 or parts[-1] in GTLDS):
        parts.pop()
    return re.sub(r"[^a-z0-9]", "", parts[-1] if parts else "")


def slugs(company: str, website: str | None) -> list[str]:
    """Candidate ATS slugs, most likely first."""
    base = re.sub(r"[^a-z0-9 ]", "", company.lower()).strip()
    out = [base.replace(" ", ""), base.replace(" ", "-")]
    first = base.split()[0] if base.split() else ""
    if first and first not in out:
        out.append(first)
    stem = domain_stem_of(website)
    if stem and stem not in out:
        out.insert(0, stem)
    return [s for s in dict.fromkeys(out) if len(s) > 2][:4]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def names_match(board_name: str | None, company: str, website: str | None = None) -> bool:
    """True when the name an ATS reports for a board is this company (either contains the other)."""
    b = _norm(board_name or "")
    return len(b) >= 4 and any(len(c) >= 4 and (c in b or b in c) for c in (_norm(company), domain_stem_of(website)))


def trust_of(slug: str, website: str | None, company: str = "", board_name: str | None = None) -> str:
    """'high' only when the board is provably this company's. A generic slug ('clarity', 'alloy', 'think')
    happily matches a different company's board, and a bare name match proves little, so:
      - the slug must be the company's domain stem, or its full squashed name when that prefixes the stem
      - if the ATS reports a board name (Greenhouse, Recruitee, Workable) it must match, else veto
      - a long multi-word name ('GRO Solutions') reported verbatim by the ATS also counts, even on a first-word slug
    """
    stem, name = domain_stem_of(website), _norm(company)
    if board_name and not names_match(board_name, company, website):
        return "UNVERIFIED-slug-guess"
    own_slug = bool(slug and stem) and (
        slug == stem
        or (_norm(slug) == name and len(min(_norm(slug), stem, key=len)) >= 6
            and (stem.startswith(_norm(slug)) or _norm(slug).startswith(stem)))
    )
    long_name_match = bool(board_name) and len(name) >= 8 and _norm(board_name) == name
    return "high" if own_slug or long_name_match else "UNVERIFIED-slug-guess"


def parse_ats_urls(text: str) -> list[tuple[str, str]]:
    """Every (board, slug) mentioned in a URL or a blob of HTML, first-seen order, generic hosts dropped."""
    found: list[tuple[str, str]] = []
    for board, pat in _ATS_PATTERNS:
        for m in pat.finditer(text):
            slug = m.group(1).lower()
            if slug not in _GENERIC_SLUGS and (board, slug) not in found:
                found.append((board, slug))
    return found
