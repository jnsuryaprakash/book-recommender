"""Ingest reading history from the user's Blogger post and enrich via Open Library.

Output: data/read.json  -- list of {title, author, ol_id, subjects, description, source}.

Skips the Telugu section (we only recommend from Plano's English digital nonfiction).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .common import DATA, env, save_json

OL_SEARCH = "https://openlibrary.org/search.json"
GBOOKS = "https://www.googleapis.com/books/v1/volumes"


def fetch_blog(url: str) -> str:
    r = httpx.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.text


def parse_entries(html: str) -> list[dict]:
    """Pulls the ordered list of 'Title - Author' entries from the post body.

    The blog post has `<strong>English:</strong>` then `<ol><li>...</li></ol>`,
    followed by `<strong>Telugu:</strong>` and another list. We keep only the
    English run by walking <li> elements that precede the Telugu marker.
    """
    tree = HTMLParser(html)
    post = tree.css_first(".post-body") or tree.css_first("body")
    if post is None:
        raise RuntimeError("Could not locate post body")

    # Find the byte offset of the Telugu marker so we can clip the html.
    # The marker may be wrapped in <strong>, <b>, or just a bare <div>Telugu:</div>.
    body_html = post.html or ""
    telugu_idx = re.search(r"\btelugu\s*:?\s*<", body_html, re.I)
    english_html = body_html[: telugu_idx.start()] if telugu_idx else body_html

    sub = HTMLParser(english_html)
    entries: list[dict] = []
    for li in sub.css("li"):
        text = li.text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).replace(" ", " ")
        if not text or len(text) > 300:
            continue
        # Strip parenthetical commentary at end (e.g. "(important book...)")
        text = re.sub(r"\s*\(.*?\)\s*$", "", text).strip()
        # Match "Title - Author" — also tolerate "Title- Author" / em dash etc.
        m = re.match(r"^(.+?)\s*[-–—]\s*(.+?)\s*$", text)
        if not m:
            continue
        title = m.group(1).strip(" .-")
        author = m.group(2).strip(" .-")
        if len(title) < 2 or len(author) < 2:
            continue
        entries.append({"title": title, "author": author})

    # De-dupe preserving order
    seen = set()
    out = []
    for e in entries:
        key = (e["title"].lower(), e["author"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def ol_lookup(title: str, author: str) -> dict | None:
    r = httpx.get(
        OL_SEARCH,
        params={"title": title, "author": author, "limit": 1},
        timeout=15,
    )
    r.raise_for_status()
    docs = r.json().get("docs", [])
    if not docs:
        return None
    d = docs[0]
    return {
        "ol_id": d.get("key", ""),
        "subjects": d.get("subject", [])[:25],
        "first_publish_year": d.get("first_publish_year"),
        "description": (d.get("first_sentence") or [""])[0] if d.get("first_sentence") else "",
    }


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def gbooks_lookup(title: str, author: str) -> dict | None:
    q = f'intitle:"{title}" inauthor:"{author}"'
    r = httpx.get(GBOOKS, params={"q": q, "maxResults": 1}, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return None
    v = items[0].get("volumeInfo", {})
    return {
        "subjects": v.get("categories", []),
        "description": v.get("description", "")[:1000],
    }


def enrich(entry: dict) -> dict:
    out = dict(entry)
    ol = None
    try:
        ol = ol_lookup(entry["title"], entry["author"])
    except Exception:
        ol = None
    if ol:
        out.update(ol)
    if not out.get("description"):
        try:
            gb = gbooks_lookup(entry["title"], entry["author"])
        except Exception:
            gb = None
        if gb:
            out["description"] = out.get("description") or gb["description"]
            if not out.get("subjects"):
                out["subjects"] = gb["subjects"]
    out.setdefault("subjects", [])
    out.setdefault("description", "")
    return out


def main():
    url = env("READS_URL")
    out_path: Path = DATA / "read.json"

    print(f"Fetching blog: {url}")
    html = fetch_blog(url)
    entries = parse_entries(html)
    print(f"Parsed {len(entries)} entries (English nonfiction).")

    enriched: list[dict] = []
    for e in tqdm(entries, desc="Enriching"):
        enriched.append(enrich(e))
        time.sleep(0.4)  # be nice to Open Library

    save_json(out_path, enriched)
    print(f"Wrote {out_path} ({len(enriched)} books)")


if __name__ == "__main__":
    main()
