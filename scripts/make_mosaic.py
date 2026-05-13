"""Download cover images for every pick and stitch them into a single mosaic PNG.

Used to embed a contact-sheet snapshot of the recommendations into the blog post.
"""
from __future__ import annotations

# truststore must be injected BEFORE ssl/httpx are imported anywhere
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import io
import ssl
from pathlib import Path

import httpx
from PIL import Image

from .common import DATA, OUTPUT, lib_data, lib_output, library_key, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", "-l", help="OverDrive library key")
    ap.add_argument("--input", type=Path, default=None, help="Override picks JSON path")
    ap.add_argument("--out", type=Path, default=None, help="Override mosaic output path")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--cover-w", type=int, default=160)
    ap.add_argument("--cover-h", type=int, default=240)
    ap.add_argument("--gap", type=int, default=6)
    args = ap.parse_args()

    lib = library_key(args.library)
    src = args.input or (lib_data(lib) / "next_100.json")
    out = args.out or (lib_output(lib) / "covers_mosaic.png")

    data = load_json(src) or {}
    picks = []
    for theme in data.get("themes", []):
        for p in theme.get("picks", []):
            if p.get("coverUrl"):
                picks.append(p)

    n = len(picks)
    cols = args.cols
    rows = (n + cols - 1) // cols
    W = cols * args.cover_w + (cols + 1) * args.gap
    H = rows * args.cover_h + (rows + 1) * args.gap

    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    print(f"Building {cols}×{rows} mosaic for {n} covers → {W}×{H}px")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with httpx.Client(timeout=20.0, follow_redirects=True, verify=ctx) as client:
        for i, p in enumerate(picks):
            r, c = divmod(i, cols)
            x = args.gap + c * (args.cover_w + args.gap)
            y = args.gap + r * (args.cover_h + args.gap)
            try:
                resp = client.get(p["coverUrl"])
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img = img.resize((args.cover_w, args.cover_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
            except Exception as e:
                print(f"  miss [{i}] {p['title']}: {e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
