# AVX Knowledge Base

This folder holds the FAA-published reference material AVX uses to answer student-pilot questions accurately.

> **Quick clarification on terminology:** Claude is a pre-trained model — we don't fine-tune it on these PDFs. Instead, when a student asks a question, we look up the most relevant passages from these documents and pass them to Claude as context along with the question. The technique is called **RAG (retrieval-augmented generation)**. The end result feels like "Claude knows the FAA materials," but mechanically Claude is reading the relevant pages each time.

## Sources

| File | What it is | Why it matters |
| --- | --- | --- |
| `14 CFR 1-59.pdf` | Title 14 of the Code of Federal Regulations, Parts 1–59 | Definitions, certification rules — Part 61 (pilots) sits in this range |
| `14 CFR 60-109.pdf` | Title 14 CFR, Parts 60–109 | Includes Part 91 (general operating rules) — *the* checkride regulation |
| `AIM 2026.pdf` | Aeronautical Information Manual, 2026 edition | Procedures, ATC phraseology, airspace, wake turbulence, etc. |
| `airplane flying handbook.pdf` | FAA-H-8083-3 Airplane Flying Handbook | Maneuvers, takeoffs/landings, emergency procedures |
| `PHAK.pdf` | FAA-H-8083-25 Pilot's Handbook of Aeronautical Knowledge | The core PPL textbook — aerodynamics, systems, weather, navigation |
| `FAA weather handbook.pdf` | FAA-H-8083-28 Aviation Weather Handbook | Weather theory and products |
| `FAA risk management Handbook.pdf` | FAA-H-8083-2 Risk Management Handbook | ADM, PAVE, IMSAFE, hazard ID |
| `PPL ACS.pdf` | Private Pilot — Airplane Airman Certification Standards | The exact tasks and standards a DPE will test on the checkride |

Total size: ~140 MB.

## Why these files are gitignored

`knowledge/*.pdf` is in `.gitignore`. Reasons:

1. **GitHub limits.** Hard 100 MB per-file cap, soft warning at 50 MB, and Render builds from git — pushing 140 MB of PDFs slows every deploy.
2. **They don't change.** These are static FAA publications. They don't belong in source-code version history.
3. **Production needs a different home.** On Render, the dyno filesystem is ephemeral on the free tier. The right pattern is to host them in object storage (S3, Cloudflare R2, Render Persistent Disk) and pull them into the build, or to ship a pre-built search index instead of the raw PDFs.

For now, keep the PDFs here locally so you can develop and test against them.

## What goes here later (when you build RAG)

A typical pipeline ends up with these subfolders:

```
knowledge/
├── *.pdf              ← raw sources (this folder, today)
├── index/             ← extracted text per source, chunked into ~500-token pieces
└── embeddings/        ← vectors for each chunk, plus a search index
```

The retrieval step at query time looks roughly like:

1. Take the student's question, embed it.
2. Find the top-K most similar chunks across all embeddings.
3. Pass those chunks + the question to Claude as context in `/api/chat`.
4. Claude answers using only what's in the prompt → answers cite real FAA sources.

A few solid library choices when you get to this stage:
- `pypdf` or `pdfplumber` for text extraction
- `tiktoken` for chunking by token count
- `voyage-3` or OpenAI `text-embedding-3-small` for embeddings (Anthropic doesn't ship its own embedding model)
- `chromadb` or `faiss` for the vector index (chromadb is friendlier for beginners)

## Source freshness

The AIM and the ACS update on a known cadence; CFR and the handbooks update more rarely. Worth re-downloading the AIM annually and re-checking the others before each deploy cycle.

- 14 CFR — https://www.ecfr.gov/current/title-14
- AIM — https://www.faa.gov/air_traffic/publications/atpubs/aim_html/
- Handbooks — https://www.faa.gov/regulations_policies/handbooks_manuals/aviation
- ACS — https://www.faa.gov/training_testing/testing/acs
