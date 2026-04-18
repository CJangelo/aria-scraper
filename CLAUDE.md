# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses `uv` for dependency management and Python 3.13+. Run all scripts with:

```bash
uv run python <script>.py
```

Install dependencies:
```bash
uv sync
```

## Pipeline Overview

This is a multi-stage RAG (retrieval-augmented generation) pipeline for extracting and querying FDA ARIA memo PDFs downloaded from the Sentinel Initiative website.

### Stage 0 — Scrape
```bash
uv run python scrape_aria_memos.py
```
Downloads all ARIA memo PDFs into `aria_memos/`.

### Stages 1–4 (per PDF) — Process

These four scripts operate on a single PDF and must run in order:

| Script | Step | Input | Output |
|---|---|---|---|
| `classify_pages.py` | 1 | `aria_memos/<drug>.pdf` | `output/<drug>/manifest.json` |
| `extract_text_pages.py` | 2 | manifest.json | `output/<drug>/pages/page_*.md` (text pages via pdfplumber) |
| `extract_image_pages.py` | 3 | manifest.json | `output/<drug>/pages/page_*.md` + `images/*.png` (OCR via RapidOCR) |
| `assemble_documents.py` | 4 | pages/*.md | `output/<drug>/final/<ref_id>_<doc_type>.md` + `final/index.json` |

Run all four steps across all PDFs at once:
```bash
uv run python batch_run.py                 # all PDFs
uv run python batch_run.py --only new      # skip already-processed
uv run python batch_run.py --only failed   # re-run failures
uv run python batch_run.py --dry-run       # preview only
```

### Stages 5–7 — Index & Query

```bash
uv run python build_drug_index.py    # aggregates all final/index.json → master_index.json
uv run python chunk_documents.py     # splits final/*.md → chunks.jsonl (375-word chunks, 50-word overlap)
uv run python embed_and_store.py     # embeds chunks.jsonl → chroma_db/ using all-MiniLM-L6-v2
uv run python retrieve.py "your query here" [-n 5] [--drug <drug_key>] [--doctype <type>]
```

## Key Data Concepts

**Sub-documents:** Each ARIA memo PDF contains multiple sub-documents (e.g., sufficiency review, clinical review) identified by a footer `Reference ID: NNNNNNN`. `classify_pages.py` detects these boundaries; `assemble_documents.py` groups pages by Reference ID into separate markdown files.

**Page types:** Pages are classified as `text` (extractable by pdfplumber) or `image` (scanned, requiring OCR). The manifest drives which extraction script handles each page.

**master_index.json:** Groups sub-documents by drug key (PDF stem, e.g. `zepbound-tirzepatide`). Drug key is more reliable than drug_name metadata for grouping.

**chunks.jsonl:** One JSON object per line. Each chunk carries metadata: `drug_key`, `drug_name`, `reference_id`, `document_type`, `source_pdf`, `page_range`.

**ChromaDB collection:** Named `aria_memos`, cosine similarity, stored at `chroma_db/`. The same `all-MiniLM-L6-v2` model must be used for both embedding and querying.

## Output Directory Structure

```
output/
  <pdf_stem>/
    manifest.json          # page classifications + pipeline metadata
    pages/
      page_001.md          # one file per extracted page
      ...
    images/
      page_130.png         # rasterized image pages
    final/
      <ref_id>_<doctype>.md  # assembled sub-documents with YAML front matter
      index.json           # sub-document index for this PDF
```

## Deprecated Code

`deprecated/classify_pages.py` contains an older page classification approach. Ignore it — the active version is at the project root.
