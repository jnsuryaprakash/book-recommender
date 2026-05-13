"""LLM re-rank with Anthropic Claude — pick 100 from ~300 with theme grouping + 'why'.

Uses Wibey's ANTHROPIC_API_KEY env var with Claude Opus 4.x.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from .common import DATA, env, lib_data, library_key, load_json, save_json

DEFAULT_MODEL = "claude-opus-4-7"  # Wibey's current opus generation


SYSTEM = """You are a thoughtful, opinionated librarian curating a personal reading queue.
Output strict JSON only — no prose, no markdown fences."""

PROMPT_TEMPLATE = """# READER TASTE PROFILE
{taste}

# TASK
From the CANDIDATE POOL below, pick EXACTLY {target} books for this reader's next reads.

# CONSTRAINTS
- Group picks into 8 to 10 themes that reflect this reader's interests.
- Each theme contains 8 to 15 picks. No theme exceeds 15.
- Maximum 3 books per author across the entire list.
- Mix proven hits with niche/under-the-radar gems.
- Prefer titles marked AVAILABLE when quality is equal, but don't sacrifice quality.
- Use ONLY titleIds from the CANDIDATE POOL. Never invent titles.
- Each pick has a one-sentence "why" tied to a specific aspect of the reader's taste.

# CANDIDATE POOL ({n} books)
{pool}

# OUTPUT JSON SCHEMA
{{
  "themes": [
    {{
      "name": "Theme title (max 5 words)",
      "blurb": "One sentence on why this theme fits the reader",
      "picks": [
        {{"titleId": "...", "title": "...", "author": "...", "why": "..."}}
      ]
    }}
  ]
}}"""


def taste_profile(reads: list[dict]) -> str:
    authors = Counter(r["author"] for r in reads if r.get("author"))
    top_authors = ", ".join(f"{a} ({n})" for a, n in authors.most_common(20))

    subj_counter: Counter = Counter()
    for r in reads:
        for s in (r.get("subjects") or [])[:6]:
            subj_counter[s.lower()] += 1
    top_subjects = ", ".join(s for s, _ in subj_counter.most_common(25))

    sample_titles = "; ".join(r["title"] for r in reads[-30:])
    return (
        f"Total books read: {len(reads)} (English nonfiction).\n"
        f"Favored authors with read counts: {top_authors}.\n"
        f"Recurring subjects: {top_subjects}.\n"
        f"Recent reads sample: {sample_titles}.\n"
        "Detected themes: leadership & management, cognitive psychology & decision making, "
        "AI/ML & technology futures, meditation & spiritual practice (Buddhist, Vedantic, Stoic), "
        "productivity & deep work, biographies of operators and founders, "
        "entrepreneurship & strategy."
    )


def candidate_block(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        avail = "AVAILABLE" if r.get("isAvailable") else f"HOLDS:{r.get('holdsCount',0)}"
        subj = (r.get("subjects") or "")[:120]
        desc = (r.get("description") or "").replace("\n", " ")[:240]
        # Strip HTML tags often present in OverDrive descriptions
        desc = re.sub(r"<[^>]+>", "", desc)
        lines.append(
            f"[{r['titleId']}] {r['title']} — {r['author']} | {avail} | subj: {subj} | {desc}"
        )
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    # Claude may wrap in ```json ... ``` despite instructions — strip it.
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", help="OverDrive library key")
    ap.add_argument("--ranked", type=Path, default=None, help="Override ranked input path")
    ap.add_argument("--reads", type=Path, default=DATA / "read.json")
    ap.add_argument("--out", type=Path, default=None, help="Override picks output path")
    ap.add_argument("--model", default=os.getenv("CHAT_MODEL", DEFAULT_MODEL))
    ap.add_argument("--pool-size", type=int, default=300)
    ap.add_argument("--target", type=int, default=100, help="Target number of picks")
    args = ap.parse_args()

    import anthropic

    lib = library_key(args.library)
    ranked_path = args.ranked or (lib_data(lib) / "ranked.json")
    out = args.out or (lib_data(lib) / "next_100.json")
    print(f"Library: {lib} — ranked={ranked_path}, out={out}")

    ranked = load_json(ranked_path) or []
    reads = load_json(args.reads) or []
    if not ranked:
        raise SystemExit(f"No ranked candidates at {ranked_path} — run rank.py first.")

    pool = ranked[: min(args.pool_size, len(ranked))]
    prompt = PROMPT_TEMPLATE.format(
        target=args.target,
        taste=taste_profile(reads),
        n=len(pool),
        pool=candidate_block(pool),
    )

    client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"))
    print(f"Re-ranking with {args.model} over {len(pool)} candidates...")
    resp = client.messages.create(
        model=args.model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    payload = extract_json(text)

    # Hydrate picks with full metadata from the ranked pool
    by_id = {str(r["titleId"]): r for r in pool}
    total = 0
    for theme in payload.get("themes", []):
        hydrated = []
        for p in theme.get("picks", []):
            meta = by_id.get(str(p.get("titleId", "")))
            if not meta:
                continue
            hydrated.append({**meta, "why": p.get("why", "")})
            total += 1
        theme["picks"] = hydrated

    save_json(out, payload)
    print(f"Wrote {out} — {total} picks across {len(payload.get('themes', []))} themes")
    print(f"Token usage: input={resp.usage.input_tokens}, output={resp.usage.output_tokens}")


if __name__ == "__main__":
    main()
