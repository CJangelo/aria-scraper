# aria-scraper

A retrieval-augmented generation (RAG) pipeline for exploring FDA ARIA (Accelerated Review of Information and Analysis) drug review memos published by the [Sentinel Initiative](https://www.sentinelinitiative.org/methods-data-tools/aria). All source documents are publicly available FDA records.

## What it does

Downloads ARIA memo PDFs, extracts and classifies their pages, assembles sub-documents (e.g. sufficiency reviews, clinical reviews) by Reference ID, chunks and embeds them into a local vector store, then lets you query across all drugs or filter by drug and document type.

## Pipeline

### Stage 0 — Scrape
```bash
uv run python scrape_aria_memos.py
```
Downloads all ARIA memo PDFs into `aria_memos/`.

### Stages 1–4 — Process PDFs

Run all four extraction steps across every PDF:
```bash
uv run python batch_run.py                 # all PDFs
uv run python batch_run.py --only new      # skip already-processed
uv run python batch_run.py --only failed   # re-run failures
uv run python batch_run.py --dry-run       # preview only
```

Or run steps individually on a single PDF:

| Script | Input | Output |
|---|---|---|
| `classify_pages.py` | `aria_memos/<drug>.pdf` | `output/<drug>/manifest.json` |
| `extract_text_pages.py` | manifest.json | `output/<drug>/pages/page_*.md` |
| `extract_image_pages.py` | manifest.json | `output/<drug>/pages/page_*.md` + `images/*.png` |
| `assemble_documents.py` | pages/*.md | `output/<drug>/final/<ref_id>_<doctype>.md` |

### Stages 5–7 — Index & Query
```bash
uv run python build_drug_index.py    # aggregates all index.json files → master_index.json
uv run python chunk_documents.py     # splits assembled docs → chunks.jsonl
uv run python embed_and_store.py     # embeds chunks → chroma_db/ (all-MiniLM-L6-v2)
uv run python retrieve.py "your query" [-n 5] [--drug <drug_key>] [--doctype <type>]
```

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Data

All documents processed by this pipeline are publicly available FDA records obtained from the [Sentinel Initiative ARIA page](https://www.sentinelinitiative.org/methods-data-tools/aria). No proprietary or non-public data is used.

## Output structure

```
output/
  <pdf_stem>/
    manifest.json          # page classifications
    pages/page_*.md        # per-page extracted text
    images/page_*.png      # rasterized image pages
    final/
      <ref_id>_<doctype>.md  # assembled sub-documents
      index.json
```
