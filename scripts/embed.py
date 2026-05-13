"""Build TF-IDF vectors for read books + catalog candidates.

We use TF-IDF locally (no API needed) — it works well when the docs are
title + author + description + subjects, and avoids any embedding-API dependency.
Vectors + raw rows are pickled for the ranker.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .common import DATA, lib_data, library_key, load_json


def doc(item: dict) -> str:
    subjects = " ".join((item.get("subjects") or [])[:10])
    desc = (item.get("description") or "")[:1200]
    subtitle = item.get("subtitle") or ""
    return f"{item.get('title','')} {subtitle} {item.get('author','')} {desc} {subjects}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", help="OverDrive library key")
    ap.add_argument("--reads", type=Path, default=DATA / "read.json")
    ap.add_argument("--catalog", type=Path, default=None, help="Override catalog path")
    ap.add_argument("--out", type=Path, default=None, help="Override pickle output path")
    args = ap.parse_args()

    lib = library_key(args.library)
    catalog_path = args.catalog or (lib_data(lib) / "catalog.json")
    pickle_path = args.out or (lib_data(lib) / "vectors.pkl")
    print(f"Library: {lib} — catalog={catalog_path}, out={pickle_path}")

    reads = load_json(args.reads) or []
    catalog = load_json(catalog_path) or []
    if not reads:
        raise SystemExit(f"No reads at {args.reads} — run ingest_reads first.")
    if not catalog:
        raise SystemExit(f"No catalog at {catalog_path} — run fetch_catalog first.")

    print(f"Reads: {len(reads)}, Catalog: {len(catalog)}")
    print("Building TF-IDF vocabulary over reads + catalog...")

    read_docs = [doc(r) for r in tqdm(reads, desc="reads")]
    cat_docs = [doc(c) for c in tqdm(catalog, desc="catalog")]

    vec = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    vec.fit(read_docs + cat_docs)
    read_vecs = vec.transform(read_docs).astype(np.float32)
    cat_vecs = vec.transform(cat_docs).astype(np.float32)
    print(f"Vocab size: {len(vec.vocabulary_)}, dims: {read_vecs.shape[1]}")

    with pickle_path.open("wb") as f:
        pickle.dump(
            {
                "vectorizer": vec,
                "read_vecs": read_vecs,
                "cat_vecs": cat_vecs,
                "reads": reads,
                "catalog": catalog,
            },
            f,
        )
    print(f"Wrote {pickle_path} ({pickle_path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
