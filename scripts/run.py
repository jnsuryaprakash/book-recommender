"""One-shot pipeline runner for a given library.

Usage:
    python -m scripts.run --library plano
    python -m scripts.run --library lib2go      # Bentonville
    python -m scripts.run --library nypl --skip-reads --skip-catalog

Stages (in order):
    1. ingest_reads       — scrape your reading list (library-independent)
    2. fetch_catalog      — walk the library's Libby catalog
    3. embed              — TF-IDF vectors
    4. rank               — k-means centroids → top 300
    5. rerank             — Claude curates the final picks
    6. render             — markdown + HTML
    7. make_mosaic        — cover contact-sheet PNG

Each stage is skippable via --skip-<stage>.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


STAGES = [
    ("reads", "ingest_reads"),
    ("catalog", "fetch_catalog"),
    ("embed", "embed"),
    ("rank", "rank"),
    ("rerank", "rerank"),
    ("render", "render"),
    ("mosaic", "make_mosaic"),
]


def run(module: str, library: str, extra: list[str]) -> None:
    # ingest_reads has no --library flag
    cmd = [sys.executable, "-m", f"scripts.{module}"]
    if module != "ingest_reads":
        cmd += ["--library", library]
    cmd += extra
    print(f"\n▶ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", required=False, default=None,
                    help="OverDrive library key (e.g. plano, lib2go)")
    for tag, _ in STAGES:
        ap.add_argument(f"--skip-{tag}", action="store_true")
    ap.add_argument("--target", type=int, default=100,
                    help="Number of picks for rerank (default: 100)")
    ap.add_argument("--pool-size", type=int, default=300)
    args = ap.parse_args()

    # Resolve library lazily — child scripts will fall back to LIBRARY_KEY env.
    library = args.library or ""

    for tag, module in STAGES:
        if getattr(args, f"skip_{tag.replace('-', '_')}"):
            print(f"⏭  skipping {module}")
            continue
        extra: list[str] = []
        if module == "rerank":
            extra += ["--target", str(args.target), "--pool-size", str(args.pool_size)]
        run(module, library, extra)

    print("\n✓ pipeline complete.")


if __name__ == "__main__":
    main()
