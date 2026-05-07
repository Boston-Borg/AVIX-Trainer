"""
Retrieval module — the "find the most relevant FAA chunks" half of RAG.

How it fits in:
  scripts/build_index.py  → builds chunks.jsonl + embeddings.npy (one time)
  retrieval.py            → loads them, answers "top-K for this question?"
  server.py               → calls retrieve() before /api/chat hits Claude

Design notes:
  - The index is loaded ONCE at import time and kept in memory. The Flask
    server imports this module at startup, so users never wait for a load.
  - If the index is missing (e.g. a fresh clone before build_index has been
    run), this module loads in "empty" mode — retrieve() returns [] and the
    server can still answer (just without FAA context).
  - We re-normalize embeddings to unit length so cosine similarity is just
    a dot product, which numpy handles in milliseconds for 6,750 rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "knowledge" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.json"

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3")


# --- One result row returned to the caller ----------------------------------
@dataclass
class Hit:
    """One retrieved chunk plus its citation info and similarity score."""
    source: str         # e.g. "PHAK.pdf"
    page_start: int
    page_end: int
    text: str
    score: float        # 0..1, higher = more similar

    def citation(self) -> str:
        """Human-readable citation string for the model to include."""
        name = self.source.replace(".pdf", "")
        if self.page_start == self.page_end:
            return f"{name} p.{self.page_start}"
        return f"{name} pp.{self.page_start}-{self.page_end}"


# --- Index loaded once at import time ---------------------------------------
class _Index:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D), L2-normalized
        self.meta: dict = {}
        self.loaded: bool = False

    def load(self) -> None:
        if self.loaded:
            return
        if not (CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists()):
            print("[retrieval] No index found at "
                  f"{INDEX_DIR}. Run scripts/build_index.py to create it.")
            return

        # chunks
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]

        # embeddings — normalize for fast cosine-as-dot-product
        raw = np.load(EMBEDDINGS_PATH).astype(np.float32)
        # Sanitize: a few chunks may have produced NaN/Inf vectors. Replace
        # them with zeros so they never match anything (they'll score 0.0).
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero
        self.embeddings = (raw / norms).astype(np.float32)

        # meta (optional)
        if META_PATH.exists():
            with META_PATH.open("r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self.loaded = True
        print(f"[retrieval] Loaded {len(self.chunks)} chunks, "
              f"vectors shape={self.embeddings.shape}")


_index = _Index()
_index.load()

# Voyage client for embedding incoming queries
_voyage_client: Optional[voyageai.Client] = (
    voyageai.Client(api_key=VOYAGE_API_KEY) if VOYAGE_API_KEY else None
)


# --- Public API -------------------------------------------------------------
def index_ready() -> bool:
    """True if the FAA index loaded successfully."""
    return _index.loaded and _index.embeddings is not None


def retrieve(query: str, top_k: int = 5, min_score: float = 0.35) -> list[Hit]:
    """Return the `top_k` most similar chunks to `query`.

    Args:
        query:     the student's question, in plain English.
        top_k:     how many chunks to return.
        min_score: filter out chunks below this cosine similarity. Stops us
                   from feeding Claude irrelevant junk when the question is
                   off-topic (e.g. "what's the weather in Paris?").

    Returns:
        A list of Hit objects, sorted by descending score. Empty list if
        the index isn't loaded or no chunks meet `min_score`.
    """
    if not index_ready():
        return []
    if _voyage_client is None:
        print("[retrieval] VOYAGE_API_KEY missing; cannot embed queries.")
        return []
    if not query.strip():
        return []

    # 1. Embed the query (note: input_type="query", not "document")
    result = _voyage_client.embed(
        texts=[query],
        model=VOYAGE_MODEL,
        input_type="query",
    )
    q = np.array(result.embeddings[0], dtype=np.float32)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q /= max(float(np.linalg.norm(q)), 1e-12)

    # 2. Cosine similarity vs. all chunks (single matmul, very fast).
    #    np.errstate silences any harmless overflow/invalid warnings from
    #    the few degenerate rows in the index — they get filtered below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = _index.embeddings @ q  # shape: (N,)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # 3. Top-K via partial sort, then full sort just on those K
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    hits: list[Hit] = []
    for idx in top_idx:
        score = float(scores[idx])
        if score < min_score:
            continue
        c = _index.chunks[idx]
        hits.append(Hit(
            source=c["source"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            text=c["text"],
            score=score,
        ))
    return hits


def format_context(hits: list[Hit], max_chars: int = 8000) -> str:
    """Format retrieved hits as a single context block to paste into Claude's
    system prompt. Each hit is wrapped with its citation tag so the model
    can quote it back."""
    if not hits:
        return ""

    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] {h.citation()}\n{h.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


# --- CLI sanity check -------------------------------------------------------
# Run `python retrieval.py "what is class B airspace"` to test from the shell.
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "what are the VFR weather minimums in class C airspace"
    print(f"Query: {q}\n")
    hits = retrieve(q, top_k=5)
    if not hits:
        print("(no hits)")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.citation()}  (score={h.score:.3f})")
        snippet = h.text.replace("\n", " ")[:240]
        print(f"    {snippet}...\n")
