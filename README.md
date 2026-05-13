# book-recommender

> Your next 100 nonfiction reads — picked from **your** library's digital shelf, grounded in **your** reading history, themed by **your** taste. Powered by Claude.

This is a personal recommender that turns the books you've already read into a curated list of 100 next reads — constrained to titles you can actually borrow from *your* OverDrive/Libby library. No "you might also like." Every pick is one click to borrow.

Two libraries are demoed in this repo out of the box:

- **Plano Public Library (TX)** — `data/plano/`, `output/plano/` — 16,450 candidates → 87 picks across 9 themes.
- **Bentonville Public Library (AR)** — `data/lib2go/`, `output/lib2go/` — 7,379 candidates → 86 picks across 9 themes.

Open `output/<library>/next_100.html` to browse the picks, or `output/<library>/covers_mosaic.png` to see the contact-sheet snapshot.

## Pipeline

```
ingest_reads → fetch_catalog → embed → rank → rerank → render → mosaic
   (blog)      (Thunder API)  (TF-IDF) (k-means) (Claude) (md+html)  (png)
```

1. **Ingest** — scrape your public reading list (Blogger format by default), enrich each title via Open Library + Google Books → `data/read.json`.
2. **Fetch catalog** — walk your library's OverDrive Thunder API for English nonfiction (ebooks + audiobooks) → `data/<lib>/catalog.json`.
3. **Embed** — local TF-IDF (sklearn, 40k vocab, 1–2-grams) over title + author + description + subjects → `data/<lib>/vectors.pkl`.
4. **Rank** — k-means(k=6) over your reading vectors, score each candidate by max cosine to any centroid, fuzzy de-dupe what you've already read, penalize long holds → `data/<lib>/ranked.json`.
5. **Re-rank** — hand the top 300 + a compact taste profile to Claude Opus 4.7; constraints: 100 picks, 8–10 themes, ≤15 per theme, ≤3 per author, one-sentence "why" tied to your actual taste → `data/<lib>/next_100.json`.
6. **Render** — Jinja2 → markdown + HTML with Libby deep links → `output/<lib>/next_100.{md,html}`.
7. **Mosaic** — stitched cover contact sheet → `output/<lib>/covers_mosaic.png`.

## Use it with your own library + Anthropic key

### 1. Clone and install

```bash
git clone https://github.com/jnsuryaprakash/book-recommender.git
cd book-recommender
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...     # https://console.anthropic.com/
LIBRARY_KEY=plano                # your OverDrive/Libby key (see below)
READS_URL=https://your.blog/your-reading-list-post   # or skip and edit ingest_reads
CHAT_MODEL=claude-opus-4-7
```

### 3. Find your library key

Open https://libbyapp.com in a browser, choose your library, and look at the URL:

```
https://libbyapp.com/library/<KEY>
```

That `<KEY>` is what goes into `LIBRARY_KEY` — or pass `--library <KEY>` to any script.

Quick verification (no key wasted):

```bash
curl -s "https://thunder.api.overdrive.com/v2/libraries/<KEY>/media?subject=111&language=en&perPage=1&page=1" \
  | python -c "import sys,json; d=json.load(sys.stdin); print('totalItems:', d.get('totalItems'))"
```

If you see a non-zero `totalItems`, you're good.

### 4. Run the whole pipeline

```bash
python -m scripts.run --library <KEY>
```

Or stage-by-stage:

```bash
python -m scripts.ingest_reads                       # ~3 min (your reading list)
python -m scripts.fetch_catalog --library <KEY>      # 2–30 min depending on catalog size
python -m scripts.embed         --library <KEY>      # ~2 min
python -m scripts.rank          --library <KEY>      # seconds
python -m scripts.rerank        --library <KEY>      # ~30 sec, one Claude call
python -m scripts.render        --library <KEY>
python -m scripts.make_mosaic   --library <KEY>
open output/<KEY>/next_100.html
```

Skip stages on re-runs:

```bash
python -m scripts.run --library <KEY> --skip-reads --skip-catalog
```

## What gets written

```
data/
  read.json                   ← your reading history (shared across libraries)
  <library>/
    catalog.json              ← every nonfiction title in that library's Libby
    vectors.pkl               ← TF-IDF vectors (gitignored; ~8–20 MB)
    ranked.json               ← top 300 candidates
    next_100.json             ← Claude's themed picks
output/
  <library>/
    next_100.md               ← human-readable themed list
    next_100.html             ← pretty version with covers + Libby links
    covers_mosaic.png         ← contact-sheet snapshot of all picks
```

## Why these choices

- **TF-IDF beats cloud embeddings here.** 210 reads + ~10k candidates is tiny. Local TF-IDF runs in ~1 second, costs $0, and doesn't leak your taste to a third party.
- **K-means(6) centroids — not a single mean.** A single centroid drowns niche interests (Krishnamurti, Adyashanti, Stoicism) under the leadership-book mass. Six centroids preserve them.
- **Claude as final curator.** Embedding similarity gives "more of the same author, ranked by popularity." A librarian doesn't do that. The LLM groups into themes, mixes hits with under-the-radar picks, and writes one-sentence "why" lines tied to your specific reads.
- **Hard borrow constraint.** Every pick is in your library's digital collection. Zero fantasy recs.

## Re-run cadence

The catalog updates weekly. Refresh monthly:

```bash
python -m scripts.run --library <KEY> --skip-reads
```

## Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED`** — corporate MITM proxy. `truststore` is imported automatically; if it still fails, install macOS root certs (`/Applications/Python\ 3.11/Install\ Certificates.command`) or run from a personal network.
- **Thunder API returns 400** — the `subject` facet uses numeric IDs (Nonfiction = 111). Format must be `ebook-overdrive` or `audiobook-overdrive`. `perPage` max is 96.
- **Claude rejects `temperature`** — already removed; Opus 4.7 doesn't accept it.
- **Re-rank picks fewer than 100** — Claude sometimes self-edits when candidate quality drops off. Bump `--pool-size 500` on `scripts.rerank` to give it more material.
- **Connection drops mid-catalog walk** — the fetcher tolerates 3 misses per format and skips. Just re-run; nothing is lost on restart.

## Adapting the reads ingester to your own source

The default `scripts/ingest_reads.py` parses a Blogger post in the format:

```html
<strong>English:</strong>
<ol>
  <li>Title – Author</li>
  ...
</ol>
```

Swap in your own source by either:

1. Editing `ingest_reads.py` to parse your HTML/CSV/Goodreads export, **or**
2. Writing `data/read.json` directly. Schema:

```json
[
  {
    "title": "Atomic Habits",
    "author": "James Clear",
    "subjects": ["self-help", "habits"],
    "description": "..."
  },
  ...
]
```

The rest of the pipeline doesn't care where reads came from.

## Future ideas

- **Hoopla as a second source** — no holds queues; great for impatient borrowing.
- **Monthly cron + email digest** of new arrivals matching your taste.
- **Thumbs up/down feedback loop** that retrains the centroids.
- **Goodreads / StoryGraph CSV ingest** for users without a blog.
- **Cross-lingual sibling** for non-English reading threads.

## Read more

The story behind this project, including the bugs that mattered and the architecture decisions, is in [`blog.md`](./blog.md).

## License

MIT — go ahead, fork it, point it at your library and your reads.
