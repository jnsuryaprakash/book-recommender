"""Walk an OverDrive (Libby) library's nonfiction catalog via the Thunder API.

Strategy: query subject=111 (Nonfiction) for ebook-overdrive + audiobook-overdrive,
paginate, dedupe by titleId, persist to data/{library}/catalog.json.

The Thunder API is undocumented but is the backbone of libbyapp.com.

Library key examples:
  --library plano       Plano Public Library (TX)
  --library lib2go      Bentonville Public Library (AR)
  --library nypl        New York Public Library
Find yours by visiting libbyapp.com/library/<key> in a browser.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .common import env, lib_data, library_key, save_json

THUNDER = "https://thunder.api.overdrive.com/v2/libraries/{lib}/media"
PER_PAGE = 96  # API max
NONFICTION_SUBJECT_ID = "111"
LANGUAGE = "en"
FORMATS = ["ebook-overdrive", "audiobook-overdrive"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (book-recommender personal use)",
    "Accept": "application/json",
    "Referer": "https://libbyapp.com/",
}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20))
def fetch_page(client: httpx.Client, lib: str, fmt: str, page: int) -> dict:
    r = client.get(
        THUNDER.format(lib=lib),
        params={
            "format": fmt,
            "subject": NONFICTION_SUBJECT_ID,
            "language": LANGUAGE,
            "perPage": PER_PAGE,
            "page": page,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def cover_url(item: dict) -> str | None:
    covers = item.get("covers") or {}
    for key in ("cover300Wide", "cover200Wide", "cover150Wide", "cover510Wide"):
        if key in covers and covers[key].get("href"):
            return covers[key]["href"]
    return None


def normalize(item: dict) -> dict:
    creators = item.get("creators") or []
    author = item.get("firstCreatorName") or (creators[0].get("name", "") if creators else "")
    subjects = [s.get("name", "") for s in (item.get("subjects") or [])]
    formats = [f.get("id") for f in (item.get("formats") or []) if f.get("id")]
    ratings = item.get("ratings") or {}
    star = (ratings.get("overallRating") or {}).get("score")
    return {
        "titleId": str(item.get("id") or ""),
        "title": (item.get("title") or "").strip(),
        "subtitle": (item.get("subtitle") or "").strip(),
        "author": author,
        "subjects": subjects,
        "description": (item.get("description") or "").strip(),
        "formats": formats,
        "isAvailable": bool(item.get("isAvailable", False)),
        "holdsCount": item.get("holdsCount", 0),
        "copiesOwned": item.get("ownedCopies", 0),
        "availableCopies": item.get("availableCopies", 0),
        "estimatedWaitDays": item.get("estimatedWaitDays"),
        "starRating": star,
        "publishDate": item.get("publishDate") or item.get("publishDateText"),
        "publisher": ((item.get("publisher") or {}) or {}).get("name", ""),
        "coverUrl": cover_url(item),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", help="OverDrive library key (overrides LIBRARY_KEY env)")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N unique titles (smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="Override output path")
    args = ap.parse_args()

    lib = library_key(args.library)
    out = args.out or (lib_data(lib) / "catalog.json")
    delay = float(env("FETCH_DELAY", "1.0"))
    print(f"Library: {lib} → {out}")

    seen: dict[str, dict] = {}
    with httpx.Client(headers=HEADERS) as client:
        for fmt in FORMATS:
            # First request to get totalItems — skip format on hard failure
            try:
                first = fetch_page(client, lib, fmt, 1)
            except Exception as exc:
                print(f"\n[{fmt}] skipped — first page failed: {exc}")
                continue
            total = first.get("totalItems", 0)
            pages = (total + PER_PAGE - 1) // PER_PAGE
            print(f"\n[{fmt}] {total} titles ({pages} pages)")

            pbar = tqdm(total=pages, desc=f"{fmt[:16]:16s}", unit="page")
            misses = 0
            for page in range(1, pages + 1):
                try:
                    data = fetch_page(client, lib, fmt, page) if page > 1 else first
                except Exception as exc:
                    tqdm.write(f"  ! page {page}: {exc}")
                    misses += 1
                    if misses >= 3:
                        tqdm.write("  giving up on this format after 3 misses")
                        break
                    time.sleep(delay * 2)
                    continue
                items = data.get("items") or []
                if not items:
                    break
                for it in items:
                    n = normalize(it)
                    if not n["titleId"]:
                        continue
                    if n["titleId"] not in seen:
                        seen[n["titleId"]] = n
                pbar.update(1)
                pbar.set_postfix(unique=len(seen))
                if args.limit and len(seen) >= args.limit:
                    break
                time.sleep(delay)
            pbar.close()
            if args.limit and len(seen) >= args.limit:
                break

    catalog = list(seen.values())
    if catalog:
        save_json(out, catalog)
    print(f"\nWrote {out} — {len(catalog)} unique titles")


if __name__ == "__main__":
    main()
