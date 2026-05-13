"""Shared helpers — paths, env, OpenAI client."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

# Use the OS keychain for SSL verification (handles corp MITM proxies cleanly).
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
DATA.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")


def env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing env var: {key}")
    return val


def library_key(cli_value: str | None = None) -> str:
    """Resolve which library to target — CLI arg > env var > default 'plano'."""
    return (cli_value or os.getenv("LIBRARY_KEY") or "plano").strip().lower()


def lib_data(lib: str) -> Path:
    """Per-library data directory: data/{lib}/."""
    p = DATA / lib
    p.mkdir(parents=True, exist_ok=True)
    return p


def lib_output(lib: str) -> Path:
    """Per-library output directory: output/{lib}/."""
    p = OUTPUT / lib
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    return _SLUG_RE.sub("-", s.lower()).strip("-")


def normalize_title(s: str) -> str:
    """For fuzzy de-dupe — lowercase, strip articles, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
