"""
Build the AVX knowledge index.

What this script does, in plain English:
  1. Walks through every PDF in the `knowledge/` folder.
  2. For each PDF, pulls the text out page by page.
  3. Splits that text into overlapping ~2,000-character chunks (so a single
     idea isn't cut in half across two chunks).
  4. Sends batches of chunks to Voyage AI, which returns a 1,024-number
     "embedding vector" for each chunk — a numerical fingerprint of meaning.
  5. Saves everything under `knowledge/index/`:
       - chunks.jsonl    : one JSON object per chunk (text + source + pages)
       - embeddings.npy  : a NumPy array of all vectors, in the same order

You only run this once (or whenever you change the PDFs). The Flask server
loads the saved files at startup and uses them to answer questions.

Run it with:
    python scripts/build_index.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pypdf
import voyageai
from dotenv import load_dotenv


# --- Configuration ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
INDEX_DIR = KNOWLEDGE_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 2000          # target characters per chunk
CHUNK_OVERLAP = 200        # characters of overlap between consecutive chunks
BATCH_SIZE = 64            # how many chunks per Voyage API call

load_dotenv(PROJECT_ROOT / ".env")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3")

if not VOYAGE_API_KEY:
    print("ERROR: VOYAGE_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)


# --- Step 1: Pull text out of one PDF, page by page -------------------------
def extract_pdf_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) for every readable page in a PDF.

    Page numbers are 1-based to match how humans (and FAA citations) count.
    Pages that fail to extract are skipped with a warning rather than crashing.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] {pdf_path.name} page {i}: extract failed ({e})")
            continue
        text = text.strip()
        if text:
            yield i, text


# --- Step 2: Cut a stream of pages into overlapping chunks ------------------
def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Combine adjacent pages, then slide a window of `chunk_size` characters
    across the combined text, advancing by (chunk_size - overlap) each step.

    For each chunk we record which pages it spans so citations can say
    "PHAK pages 12–13" instead of just "somewhere in PHAK".
    """
    if not pages:
        return []

    # Build one big string with markers so we can map every character back
    # to a page number.
    full_text_parts: list[str] = []
    char_to_page: list[int] = []
    for page_num, text in pages:
        full_text_parts.append(text)
        char_to_page.extend([page_num] * len(text))
        # Add a separator between pages (a newline counts as one char on a
        # virtual page; use the previous page so it doesn't drift).
        full_text_parts.append("\n")
        char_to_page.append(page_num)

    full_text = "".join(full_text_parts)
    chunks: list[dict] = []

    step = chunk_size - overlap
    for start in range(0, len(full_text), step):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end].strip()
        if len(chunk_text) < 100:  # skip tiny scraps (front matter, etc.)
            if end >= len(full_text):
                break
            continue

        page_start = char_to_page[start]
        page_end = char_to_page[min(end - 1, len(char_to_page) - 1)]
        chunks.append({
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
        })

        if end >= len(full_text):
            break

    return chunks


# --- Step 3: Get embeddings for a list of chunks ----------------------------
def embed_chunks(client: voyageai.Client, chunks: list[dict]) -> np.ndarray:
    """Call Voyage AI in batches and stack all embeddings into one matrix."""
    all_vectors: list[list[float]] = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch = chunks[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Voyage occasionally throws transient errors; retry up to 3 times.
        for attempt in range(3):
            try:
                result = client.embed(
                    texts=texts,
                    model=VOYAGE_MODEL,
                    input_type="document",  # use "query" at search time
                )
                all_vectors.extend(result.embeddings)
                break
            except Exception as e:  # noqa: BLE001
                wait = 2 ** attempt
                print(f"    [warn] batch {batch_idx + 1}/{total_batches} failed "
                      f"({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding batch {batch_idx + 1} failed after retries")

        done = min((batch_idx + 1) * BATCH_SIZE, len(chunks))
        print(f"    embedded {done}/{len(chunks)} chunks "
              f"(batch {batch_idx + 1}/{total_batches})")

    return np.array(all_vectors, dtype=np.float32)


# --- Main -------------------------------------------------------------------
def main() -> None:
    pdfs = sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {KNOWLEDGE_DIR}. Put your FAA PDFs there first.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs in {KNOWLEDGE_DIR}")
    for p in pdfs:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name}  ({size_mb:.1f} MB)")
    print()

    # ---- Extract + chunk -------------------------------------------------
    all_chunks: list[dict] = []
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        t0 = time.time()
        pages = list(extract_pdf_pages(pdf))
        chunks = chunk_pages(pages, source=pdf.name)
        all_chunks.extend(chunks)
        print(f"  → {len(pages)} pages, {len(chunks)} chunks "
              f"({time.time() - t0:.1f}s)")
    print()
    print(f"Total chunks across all PDFs: {len(all_chunks)}")
    print()

    # ---- Embed -----------------------------------------------------------
    print(f"Embedding with Voyage model: {VOYAGE_MODEL}")
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    t0 = time.time()
    vectors = embed_chunks(client, all_chunks)
    print(f"Embedding complete in {time.time() - t0:.1f}s "
          f"(shape: {vectors.shape})")
    print()

    # ---- Save ------------------------------------------------------------
    chunks_path = INDEX_DIR / "chunks.jsonl"
    embeddings_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.json"

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    np.save(embeddings_path, vectors)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": VOYAGE_MODEL,
            "dim": int(vectors.shape[1]),
            "num_chunks": int(vectors.shape[0]),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "sources": sorted({c["source"] for c in all_chunks}),
        }, f, indent=2)

    print(f"Wrote {chunks_path}")
    print(f"Wrote {embeddings_path}  ({embeddings_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Wrote {meta_path}")
    print()
    print("Index built. You're ready for Step 4d.")


if __name__ == "__main__":
    main()
