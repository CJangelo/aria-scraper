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
                # Replace newlines within cells with spaces
                cell = str(row[i]).replace("\n", " ").strip()
                # Escape pipe characters so they don't break the table
                cell = cell.replace("|", "\\|")
            else:
                cell = ""
            clean_row.append(cell)
        cleaned.append(clean_row)

    if not cleaned:
        return ""

    # Build the Markdown table
    lines = []

    # Header row (first row of the table)
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")

    # Separator row
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Data rows
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

    Strategy:
      1. Find all tables and their bounding boxes.
      2. Extract full-page text, excluding table regions, to get prose.
      3. Determine vertical position of each content block (prose chunk
         or table) and output them in top-to-bottom order.
    """
    # --- Find tables ---
    tables = page.find_tables()
    table_data = []
    for table in tables:
        bbox = table.bbox  # (x0, top, x1, bottom)
        cells = table.extract()
        md = table_to_markdown(cells)
        if md:
            # Use the top y-coordinate for ordering
            table_data.append({"y_top": bbox[1], "type": "table", "content": md})

    # --- Extract prose (text outside table regions) ---
    if tables:
        # Crop out table regions to get only prose text
        # Build a list of table bounding boxes
        table_bboxes = [t.bbox for t in tables]

        # Strategy: extract text from the full page, then extract text
        # from each table bbox, and subtract.  But this loses ordering.
        #
        # Better: split the page into horizontal bands between tables
        # and extract text from each band.

        # Collect all y-boundaries (top of page, top/bottom of each table, bottom of page)
        y_boundaries = [0]  # top of page
        for bbox in sorted(table_bboxes, key=lambda b: b[1]):
            y_boundaries.append(bbox[1])  # top of table
            y_boundaries.append(bbox[3])  # bottom of table
        y_boundaries.append(page.height)  # bottom of page

        # Extract text from each non-table band
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
        # No tables — extract the full page as prose
        text = page.extract_text()
        if text and text.strip():
            return text.strip()
        return ""

    # --- Merge and sort all blocks by vertical position ---
    all_blocks = prose_blocks + table_data
    all_blocks.sort(key=lambda b: b["y_top"])

    # --- Build final Markdown ---
    parts = []
    for block in all_blocks:
        if block["type"] == "table":
            parts.append("")  # blank line before table
            parts.append(block["content"])
            parts.append("")  # blank line after table
        else:
            parts.append(block["content"])

    return "\n\n".join(part for part in parts if part is not None)


def clean_page_text(text: str) -> str:
    """Light cleanup of extracted text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Don't strip meaningful indentation, but clean trailing whitespace
        cleaned.append(line.rstrip())

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_text_pages.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    # --- Load manifest ---
    out_dir = OUTPUT_ROOT / pdf_path.stem
    manifest_path = out_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        print("Run classify_pages.py first.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # --- Set up output directory ---
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # --- Identify text pages ---
    text_pages = [p for p in manifest["pages"] if p["page_type"] == "text"]
    image_pages = [p for p in manifest["pages"] if p["page_type"] == "image"]

    print(f"Extracting {len(text_pages)} text pages from {pdf_path.name}")
    print(f"  (skipping {len(image_pages)} image pages for Script 3)")
    print()

    # --- Open PDF and process ---
    pdf = pdfplumber.open(str(pdf_path))
    t0 = time.time()

    extracted = 0
    skipped_sig = 0
    skipped_cover = 0
    errors = 0

    for page_info in text_pages:
        page_num = page_info["page_num"]  # 1-indexed
        ref_id = page_info["reference_id"] or "unknown"
        has_tables = page_info["has_tables"]
        is_sig = page_info["is_signature_page"]
        is_cover = page_info["is_cover_page"]

        # --- Extract content ---
        try:
            page = pdf.pages[page_num - 1]  # 0-indexed
            raw_content = extract_text_with_tables(page)
            content = clean_page_text(raw_content)
        except Exception as exc:
            print(f"  ERROR on page {page_num}: {exc}")
            errors += 1
            continue

        if not content.strip():
            continue

        # --- Build page Markdown with metadata header ---
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

        # --- Write to file ---
        page_file = pages_dir / f"page_{page_num:03d}.md"
        page_file.write_text(md_text, encoding="utf-8")
        extracted += 1

        # Progress
        if extracted % 25 == 0:
            print(f"  … {extracted} pages extracted")

    pdf.close()
    elapsed = time.time() - t0

    # --- Update manifest with extraction status ---
    manifest["text_extraction"] = {
        "pages_extracted": extracted,
        "errors": errors,
        "output_dir": str(pages_dir),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"TEXT EXTRACTION COMPLETE")
    print(f"  Pages extracted:   {extracted}")
    print(f"  Errors:            {errors}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Output:            {pages_dir}")
    print()
    print("Next step: spot-check a few .md files in the pages/ folder,")
    print("then run extract_image_pages.py for the image pages.")
    print("=" * 60)


if __name__ == "__main__":
    main()
