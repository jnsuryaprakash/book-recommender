"""Score catalog candidates against the reader's TF-IDF taste centroids.

Strategy:
 - Build k-means(6) centroids over the reader's vectors so niche interests survive
   (Krishnamurti, Adyashanti, Stoicism, etc. don't get averaged out by leadership books).
 - Score each candidate = max cosine similarity across centroids
   minus a small penalty for already-read fuzzy matches and long hold waits.
 - Hard-filter already-read titles via rapidfuzz on normalized title strings.
 - Emit data/ranked.json with the top N candidates for the LLM re-ranker.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as l2_normalize

from .common import lib_data, library_key, normalize_title, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", help="OverDrive library key")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--pickle", type=Path, default=None, help="Override pickle input path")
    ap.add_argument("--out", type=Path, default=None, help="Override ranked output path")
    args = ap.parse_args()

    lib = library_key(args.library)
    pickle_path = args.pickle or (lib_data(lib) / "vectors.pkl")
    out = args.out or (lib_data(lib) / "ranked.json")
    print(f"Library: {lib} — pickle={pickle_path}, out={out}")

    if not pickle_path.exists():
        raise SystemExit(f"Missing {pickle_path} — run embed.py first.")
    with pickle_path.open("rb") as f:
        bundle = pickle.load(f)

    reads = bundle["reads"]
    catalog = bundle["catalog"]
    read_vecs = l2_normalize(bundle["read_vecs"])
    cat_vecs = l2_normalize(bundle["cat_vecs"])
    print(f"Reads: {len(reads)}, Catalog: {len(catalog)}")

    # Cluster the read embeddings to keep niche interests alive
    k = min(args.clusters, max(2, len(reads) // 8))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(read_vecs.toarray())
    centroids = l2_normalize(km.cluster_centers_)
    print(f"K-means clusters: {k}")

    # Cosine sim (sparse @ dense.T → dense ndarray/matrix)
    sims = cat_vecs @ centroids.T
    sims = np.asarray(sims)
    # Each candidate scored by best-fitting centroid
    scores = sims.max(axis=1)

    # Hard-filter already-read titles
    read_norm = [normalize_title(r["title"]) for r in reads]
    read_norm_set = set(read_norm)

    rows = []
    for i, cand in enumerate(catalog):
        nt = normalize_title(cand["title"])
        if not nt:
            continue
        if nt in read_norm_set:
            continue
        match = process.extractOne(nt, read_norm, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 88:
            continue

        wait_penalty = 0.0
        copies = cand.get("copiesOwned") or 0
        holds = cand.get("holdsCount") or 0
        if copies > 0 and holds > 0:
            ratio = holds / copies
            wait_penalty = min(0.05, 0.01 * ratio)

        # Tiny boost for higher-rated books to break ties
        star = cand.get("starRating") or 0
        star_boost = float(star) * 0.002

        score = float(scores[i]) - wait_penalty + star_boost
        rows.append(
            {
                "titleId": cand["titleId"],
                "title": cand["title"],
                "author": cand["author"],
                "subjects": ", ".join((cand.get("subjects") or [])[:8]),
                "description": (cand.get("description") or "")[:600],
                "score": round(score, 5),
                "isAvailable": cand.get("isAvailable", False),
                "holdsCount": holds,
                "copiesOwned": copies,
                "estimatedWaitDays": cand.get("estimatedWaitDays"),
                "starRating": star,
                "coverUrl": cand.get("coverUrl"),
                "publisher": cand.get("publisher", ""),
                "publishDate": cand.get("publishDate"),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[: args.top]
    save_json(out, top)
    print(f"Wrote {out} ({len(top)} candidates)")
    print("Top 10 preview:")
    for r in top[:10]:
        print(f"  {r['score']:.3f}  {r['title']}  —  {r['author']}")


if __name__ == "__main__":
    main()
