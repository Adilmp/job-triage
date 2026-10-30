"""Text cleanup and the two hashes.

content_hash answers "was this post edited?" (title|location|description). fingerprint answers "is this the
same role on another board?" (company|title). They are different questions, hence two hashes (P6).
"""
import hashlib
import html
import re
import unicodedata

_DROP_BLOCKS = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_LI_OPEN = re.compile(r"<li\b[^>]*>", re.I)
_LIST_OPEN = re.compile(r"<(ul|ol)\b[^>]*>", re.I)
_LINE_BREAK = re.compile(r"<br\s*/?>|</(li|tr)\s*>", re.I)
_PARA_BREAK = re.compile(r"</(p|div|ul|ol|h[1-6]|table|section|article)\s*>", re.I)
_TAG = re.compile(r"<[^>]+>")


def strip_html(raw: str | None) -> str:
    """HTML or HTML-escaped HTML to plain text, keeping paragraph and list structure.

    Greenhouse (content=true) returns its HTML *escaped*, so we unescape first; without that the model only
    sees the title (P1). Entities left behind by the tags (&amp;nbsp; -> &nbsp;) are unescaped a second time.
    """
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _DROP_BLOCKS.sub(" ", text)
    text = _LI_OPEN.sub("- ", _LIST_OPEN.sub("\n", text))
    text = _LINE_BREAK.sub("\n", text)
    text = _PARA_BREAK.sub("\n\n", text)
    return tidy_text(html.unescape(_TAG.sub(" ", text)))


def tidy_text(text: str | None) -> str:
    """Plain text with NFKC, no nbsp, single spaces, and at most one blank line between paragraphs."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[^\S\n]+", " ", text)           # runs of spaces/tabs/nbsp -> one space
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def content_hash(title: str, location: str | None, description_text: str) -> str:
    key = "|".join(_norm(p) for p in (title, location, description_text))
    return hashlib.sha256(key.encode()).hexdigest()


_LEGAL_SUFFIXES = {
    "pty", "ltd", "limited", "llc", "llp", "inc", "incorporated", "corp", "corporation", "co", "plc",
    "gmbh", "ag", "bv", "sa", "fze", "fzco", "fz", "pvt", "private",
}
_TITLE_WORDS = {
    "sr": "senior", "jr": "junior", "eng": "engineer", "engg": "engineer", "mgr": "manager",
    "dev": "developer", "swe": "software engineer",
}
_GENDER_TAG = re.compile(r"\(?\b[mfwdx]\s*/\s*[mfwdx](\s*/\s*[mfwdx])?\b\)?", re.I)   # "(m/f/d)" on EU boards


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    return re.sub(r"[^\w\s]", " ", text).split()


def normalize_company(name: str) -> str:
    toks = _tokens(name)
    while toks and toks[-1] in _LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def normalize_title(title: str) -> str:
    title = _GENDER_TAG.sub(" ", unicodedata.normalize("NFKC", title))
    return " ".join(w for t in _tokens(title) for w in _TITLE_WORDS.get(t, t).split())


def fingerprint(company: str, title: str) -> str:
    return hashlib.sha256(f"{normalize_company(company)}|{normalize_title(title)}".encode()).hexdigest()
