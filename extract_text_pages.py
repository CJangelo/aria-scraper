"""
Script 2 of 4 — extract_text_pages.py

Read the manifest from Script 1.  For every page classified as "text",
extract prose and tables using pdfplumber and write one Markdown file
per page into a staging directory.

Tables are converted to Markdown tables inline with the surrounding text,
preserving document flow.  Pages classified as "image" are skipped (those
are handled by Script 3).

Usage:
    uv run python extract_text_pages.py <path_to_pdf>
    uv run python extract_text_pages.py aria_memos/zepbound-tirzepatide.pdf

Reads:
    output/<pdf_stem>/manifest.json

Writes:
    output/<pdf_stem>/pages/page_001.md
    output/<pdf_stem>/pages/page_002.md
    ...
"""

import json
import sys
import time
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("output")


# ---------------------------------------------------------------------------
# Table → Markdown conversion
# ---------------------------------------------------------------------------

def table_to_markdown(table_data: list[list]) -> str:
    """
    Convert a pdfplumber table (list of lists) to a Markdown table string.
    Handles None cells, inconsistent column counts, and messy whitespace.
    """
    if not table_data or len(table_data) < 1:
        return ""

    # Determine max columns across all rows
    max_cols = max(len(row) for row in table_data)

    # Clean and normalize every cell
    cleaned = []
    for row in table_data:
        clean_row = []
        for i in range(max_cols):
            if i < len(row) and row[i] is not None:
                cell = str(row[i]).replace("\n", " ").strip()
                cell = cell.replace("|", "\\|")
            else:
                cell = ""
            clean_row.append(cell)
        cleaned.append(clean_row)

    if not cleaned:
        return ""

    lines = []
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_text_with_tables(page) -> str:
    """
    Extract text and tables from a pdfplumber page, preserving their
    vertical order in the document flow.
    """
    # --- Find tables ---
    tables = page.find_tables()
    table_data = []
    for table in tables:
        bbox = table.bbox
        cells = table.extract()
        md = table_to_markdown(cells)
        if md:
            table_data.append({"y_top": bbox[1], "type": "table", "content": md})

    # --- Extract prose (text outside table regions) ---
    if tables:
        table_bboxes = [t.bbox for t in tables]

        y_boundaries = [0]
        for bbox in sorted(table_bboxes, key=lambda b: b[1]):
            y_boundaries.append(bbox[1])
            y_boundaries.append(bbox[3])
        y_boundaries.append(page.height)

        prose_blocks = []
        for i in range(0, len(y_boundaries) - 1, 2):
            band_top = y_boundaries[i]
            band_bottom = y_boundaries[i + 1] if i + 1 < len(y_boundaries) else page.height

            if band_bottom - band_top < 5:
                continue

            try:
                cropped = page.crop((0, band_top, page.width, band_bottom))
                text = cropped.extract_text()
                if text and text.strip():
                    prose_blocks.append({
                        "y_top": band_top,
                        "type": "prose",
                        "content": text.strip(),
                    })
            except Exception:
                pass
    else:
        text = page.extract_text()
        if text and text.strip():
            return text.strip()
        return ""

    all_blocks = prose_blocks + table_data
    all_blocks.sort(key=lambda b: b["y_top"])

    parts = []
    for block in all_blocks:
        if block["type"] == "table":
            parts.append("")
            parts.append(block["content"])
            parts.append("")
        else:
            parts.append(block["content"])

    return "\n\n".join(part for part in parts if part is not None)


def clean_page_text(text: str) -> str:
    """Light cleanup of extracted text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        cleaned.append(line.rstrip())
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# run() — importable entry point for batch processing
# ---------------------------------------------------------------------------

def run(pdf_path: Path) -> dict:
    """
    Extract all text pages for a PDF.  Returns a result dict:
        {"status": "ok", "pages_extracted": int, "errors": int, ...}
    Raises on fatal errors (missing manifest, missing PDF).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = OUTPUT_ROOT / pdf_path.stem
    manifest_path = out_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run classify_pages.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    text_pages = [p for p in manifest["pages"] if p["page_type"] == "text"]
    image_pages = [p for p in manifest["pages"] if p["page_type"] == "image"]

    print(f"Extracting {len(text_pages)} text pages from {pdf_path.name}")
    print(f"  (skipping {len(image_pages)} image pages for Script 3)")

    pdf = pdfplumber.open(str(pdf_path))
    t0 = time.time()

    extracted = 0
    errors = 0

    for page_info in text_pages:
        page_num = page_info["page_num"]
        ref_id = page_info["reference_id"] or "unknown"
        has_tables = page_info["has_tables"]
        is_sig = page_info["is_signature_page"]
        is_cover = page_info["is_cover_page"]

        try:
            page = pdf.pages[page_num - 1]
            raw_content = extract_text_with_tables(page)
            content = clean_page_text(raw_content)
        except Exception as exc:
            print(f"  ERROR on page {page_num}: {exc}")
            errors += 1
            continue

        if not content.strip():
            continue

        md_lines = [
            f"<!-- page: {page_num} | ref_id: {ref_id} "
            f"| has_tables: {has_tables} "
            f"| is_signature: {is_sig} "
            f"| is_cover: {is_cover} -->",
            "",
            content,
            "",
        ]

        md_text = "\n".join(md_lines)
        page_file = pages_dir / f"page_{page_num:03d}.md"
        page_file.write_text(md_text, encoding="utf-8")
        extracted += 1

        if extracted % 25 == 0:
            print(f"  … {extracted} pages extracted")

    pdf.close()
    elapsed = time.time() - t0

    # Update manifest
    manifest["text_extraction"] = {
        "pages_extracted": extracted,
        "errors": errors,
        "output_dir": str(pages_dir),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "pages_extracted": extracted,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_text_pages.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    try:
        result = run(pdf_path)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"TEXT EXTRACTION COMPLETE")
    print(f"  Pages extracted:   {result['pages_extracted']}")
    print(f"  Errors:            {result['errors']}")
    print(f"  Time:              {result['elapsed_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
