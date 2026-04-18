# aria-rag

A retrieval-augmented generation (RAG) system for querying FDA drug safety reviews. Ask natural language questions across ARIA (Accelerated Review of Information and Analysis) memos published by the [Sentinel Initiative](https://www.sentinelinitiative.org/methods-data-tools/aria) — covering sufficiency reviews, clinical reviews, and other sub-documents for dozens of drugs. All source documents are publicly available FDA records.

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
uv run python embed_and_store.py     # embeds new chunks → chroma_db/, removes stale ones
```

### Querying

**`query.py` — retrieval + LLM synthesis (recommended)**

```bash
uv run python query.py "what are the safety concerns for tirzepatide"
uv run python query.py "..." --drug zepbound          # filter by drug (partial match)
uv run python query.py "..." --doctype aria_sufficiency  # filter by document type
uv run python query.py "..." -n 10                    # retrieve more chunks
uv run python query.py "..." --no-generate            # skip LLM, print raw chunks only
```

The model defaults to `claude-haiku-4-5-20251001`. Override with `--model` or the `LLM_MODEL` environment variable. LiteLLM is used under the hood, so any compatible model string works:

```bash
uv run python query.py "..." --model claude-sonnet-4-6   # Anthropic
uv run python query.py "..." --model gpt-4o              # OpenAI
uv run python query.py "..." --model ollama/llama3       # local via Ollama
```

Set the appropriate API key as an environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) before running.

**`retrieve.py` — retrieval only**

```bash
uv run python retrieve.py "your query" [-n 5] [--drug <name>] [--doctype <type>]
```

Returns raw chunks from ChromaDB without LLM synthesis. Useful for inspecting what the vector store is returning.

### Diagnostics

After running `batch_run.py`, inspect extraction quality with:

```bash
uv run python diagnostics.py            # summary: page counts, errors, timing
uv run python diagnostics.py --detail   # per-PDF breakdown table
uv run python diagnostics.py --csv      # write diagnostics.csv
```

Reads `batch_results.json` plus each PDF's `manifest.json` and `index.json` to report text vs. image page counts, extraction errors, sub-document counts, and elapsed time.

## Setup

Everything uses `uv`. No pip, no manual venvs, no conda.

### One-time installation

```powershell
# Allow PowerShell scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Close and reopen PowerShell, then install Python
uv python install 3.13
```

### Clone and install dependencies

```powershell
cd path\to\aria-scraper
uv sync
```

### Opening the project in VSCode

Open VSCode. **File → Open Folder** → navigate to the project folder → **Select Folder**. Open the terminal: **Terminal → New Terminal**. VSCode will detect the `.venv` and set up the interpreter automatically.

To make this faster, save the workspace: **File → Save Workspace As** → name it. Next time, double-click the `.code-workspace` file to open everything at once.

If VSCode auto-activated an old venv (you'll see a prefix like `(something)` in your prompt), clear it once:

```powershell
deactivate
```

Then ignore activation forever — `uv run` handles it.

### Running scripts

```powershell
uv run python my_script.py       # run a script
uv run python                    # interactive REPL
```

### Adding or removing packages

```powershell
uv add requests                  # install + record in pyproject.toml
uv add --dev ruff pytest         # dev-only dependencies
uv remove some-package           # uninstall
```

### Reproducing the environment on another machine

```powershell
uv sync
```

### Check if a package is installed

```powershell
uv run python -c "import PACKAGE_NAME; print(PACKAGE_NAME.__version__)"
```

### Quick reference

| Task                     | Command                                          |
|--------------------------|--------------------------------------------------|
| New project              | `uv init my-project`                             |
| Pin Python version       | `uv python pin 3.13`                             |
| Add a package            | `uv add package-name`                            |
| Run a script             | `uv run python script.py`                        |
| Reproduce environment    | `uv sync`                                        |
| See installed packages   | `uv pip list`                                    |
| Upgrade one package      | `uv lock --upgrade-package X && uv sync`         |
| Upgrade all              | `uv lock --upgrade && uv sync`                   |

### Common package names

| Install with              | Import as                                | What it does                    |
|---------------------------|------------------------------------------|---------------------------------|
| `pymupdf`                 | `import fitz`                            | PDF text extraction             |
| `pdfplumber`              | `import pdfplumber`                      | PDF tables + layout-aware text  |
| `beautifulsoup4`          | `from bs4 import BeautifulSoup`          | HTML parsing                    |
| `requests`                | `import requests`                        | HTTP requests                   |
| `sentence-transformers`   | `from sentence_transformers import ...`  | Embeddings                      |
| `chromadb`                | `import chromadb`                        | Vector store for RAG            |
| `Pillow`                  | `from PIL import Image`                  | Image processing                |

### Gotchas

- **Two Documents folders on Windows:** `~\Documents` may resolve to your local Documents folder instead of the OneDrive-synced one. Use the full path to avoid ambiguity.
- **VSCode auto-activates old venvs:** Run `deactivate` once if you see an unexpected prefix in your prompt. Then forget about activation — `uv run` handles it.
- **Package name ≠ import name:** See the table above. This is just how Python works.

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

## Roadmap

### Query experience
- Interactive query loop — load the embedding model once, run multiple questions without restarting
- Partial matching for `--doctype` filter (currently requires exact string)
- Simple web UI (e.g. Streamlit) for non-CLI use

### Expand data coverage
Currently the pipeline scrapes one PDF per drug — the ARIA memo. Each drug page on the Sentinel Initiative also links to an **FDA Approval Package** on accessdata.fda.gov, which contains the full NDA/BLA review package: medical reviews, statistical reviews, labeling documents, and more.

Next step: for each drug, follow the approval package link, scrape all sub-documents from that table of contents, and run them through the same extraction and embedding pipeline. This will significantly expand the safety information available to the RAG.

### Infrastructure
- Automated tests for extraction and retrieval
- Scheduled re-scrape to pick up newly published ARIA memos
