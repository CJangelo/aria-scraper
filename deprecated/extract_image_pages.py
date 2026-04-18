"""
Script 3 of 4 — extract_image_pages.py

Read the manifest from Script 1.  For every page classified as "image",
rasterize it, run RapidOCR, and reconstruct tables from bounding box
coordinates.

RapidOCR uses PaddlePaddle's PP-OCRv4 model (packaged as ONNX), which
handles low-contrast backgrounds and FDA redaction labels far better
than Tesseract, docling, marker-pdf, or EasyOCR.

The key challenge is that OCR returns a flat list of text fragments with
bounding boxes.  We reconstruct table structure by:
  1. Grouping fragments into rows (similar y-coordinates)
  2. Sorting columns by x-coordinate within each row
  3. Detecting whether the page is a table or prose based on alignment

Usage:
    uv run python extract_image_pages.py <path_to_pdf>
    uv run python extract_image_pages.py zepbound-tirzepatide.pdf

Reads:
    output/<pdf_stem>/manifest.json

Writes:
    output/<pdf_stem>/pages/page_130.md
    output/<pdf_stem>/images/page_130.png

Prerequisites:
    uv add rapidocr-onnxruntime Pillow numpy
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("Missing RapidOCR. Run:")
    print("  uv add rapidocr-onnxruntime")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("output")
RASTER_DPI = 300

# Row grouping: fragments within this many pixels vertically are the same row
ROW_TOLERANCE = 15

# Column detection: if most rows have similar fragment counts and x-positions
# align across rows, it's a table.  Otherwise treat as prose.
TABLE_COLUMN_THRESHOLD = 3  # minimum columns to consider it a table


# ---------------------------------------------------------------------------
# Rasterize
# ---------------------------------------------------------------------------

def rasterize_page(doc: fitz.Document, page_num: int,
                   dpi: int = 300) -> Image.Image:
    """Rasterize a single PDF page (1-indexed) to a PIL Image."""
    page = doc[page_num - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_ocr(engine: RapidOCR, img: Image.Image) -> list[dict]:
    """
    Run RapidOCR on an image.  Returns a list of dicts:
        {"text": str, "confidence": float, "bbox": [x0, y0, x1, y1]}

    The raw RapidOCR output has bounding boxes as 4 corner points.
    We convert to simple [x0, y0, x1, y1] rectangles.
    """
    img_array = np.array(img)
    result, _ = engine(img_array)

    if not result:
        return []

    fragments = []
    for detection in result:
        bbox_points, text, confidence = detection

        # bbox_points is [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
        # Convert to simple rectangle
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        fragments.append({
            "text": text.strip(),
            "confidence": confidence,
            "bbox": [x0, y0, x1, y1],
            "y_center": (y0 + y1) / 2,
            "x_center": (x0 + x1) / 2,
        })

    return fragments


# ---------------------------------------------------------------------------
# Table reconstruction
# ---------------------------------------------------------------------------

def group_into_rows(fragments: list[dict],
                    tolerance: int = ROW_TOLERANCE) -> list[list[dict]]:
    """
    Group OCR fragments into rows based on y-coordinate proximity.
    Fragments whose y_center values are within `tolerance` pixels
    are considered part of the same row.
    """
    if not fragments:
        return []

    # Sort by y_center
    sorted_frags = sorted(fragments, key=lambda f: f["y_center"])

    rows = []
    current_row = [sorted_frags[0]]
    current_y = sorted_frags[0]["y_center"]

    for frag in sorted_frags[1:]:
        if abs(frag["y_center"] - current_y) <= tolerance:
            current_row.append(frag)
        else:
            # Sort the completed row by x-coordinate
            current_row.sort(key=lambda f: f["x_center"])
            rows.append(current_row)
            current_row = [frag]
            current_y = frag["y_center"]

    # Don't forget the last row
    current_row.sort(key=lambda f: f["x_center"])
    rows.append(current_row)

    return rows


def detect_column_positions(rows: list[list[dict]],
                            tolerance: int = 40) -> list[float] | None:
    """
    Try to detect consistent column positions across rows.
    If fragments align vertically across multiple rows, we have a table.

    Returns a sorted list of column x-positions, or None if no table
    structure is detected.
    """
    if len(rows) < 3:
        return None

    # Collect all x_center values from all fragments
    all_x_centers = []
    for row in rows:
        for frag in row:
            all_x_centers.append(frag["x_center"])

    if not all_x_centers:
        return None

    # Cluster x-positions to find column centers
    all_x_centers.sort()
    columns = []
    current_cluster = [all_x_centers[0]]

    for x in all_x_centers[1:]:
        if x - current_cluster[-1] <= tolerance:
            current_cluster.append(x)
        else:
            columns.append(np.mean(current_cluster))
            current_cluster = [x]
    columns.append(np.mean(current_cluster))

    # Only consider it a table if enough columns are detected
    if len(columns) < TABLE_COLUMN_THRESHOLD:
        return None

    # Verify that multiple rows use these columns
    # (at least half of rows should have fragments near column positions)
    rows_matching = 0
    for row in rows:
        row_x_centers = [f["x_center"] for f in row]
        matches = 0
        for col_x in columns:
            if any(abs(rx - col_x) < tolerance for rx in row_x_centers):
                matches += 1
        if matches >= len(columns) * 0.4:
            rows_matching += 1

    if rows_matching < len(rows) * 0.3:
        return None

    return columns


def assign_to_columns(row: list[dict], columns: list[float],
                      tolerance: int = 60) -> list[str]:
    """
    Assign each fragment in a row to the nearest column position.
    Returns a list of cell values aligned to the column positions.
    """
    cells = [""] * len(columns)

    for frag in row:
        # Find the nearest column
        distances = [abs(frag["x_center"] - col_x) for col_x in columns]
        nearest_col = int(np.argmin(distances))

        # If there's already content in this cell, append with space
        if cells[nearest_col]:
            cells[nearest_col] += " " + frag["text"]
        else:
            cells[nearest_col] = frag["text"]

    return cells


def rows_to_markdown_table(rows: list[list[dict]],
                           columns: list[float]) -> str:
    """
    Convert grouped rows into a Markdown table using detected column
    positions.
    """
    if not rows or not columns:
        return ""

    md_rows = []

    for row in rows:
        cells = assign_to_columns(row, columns)
        md_rows.append(cells)

    if not md_rows:
        return ""

    # Build markdown
    lines = []

    # First row as header
    header = md_rows[0]
    lines.append("| " + " | ".join(c if c else "" for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Remaining rows
    for row_cells in md_rows[1:]:
        # Escape pipe characters
        cleaned = [c.replace("|", "\\|") for c in row_cells]
        lines.append("| " + " | ".join(cleaned) + " |")

    return "\n".join(lines)


def rows_to_prose(rows: list[list[dict]]) -> str:
    """
    When no table structure is detected, output as plain text
    with one line per row.
    """
    lines = []
    for row in rows:
        line_text = " ".join(f["text"] for f in row)
        if line_text.strip():
            lines.append(line_text.strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page processing
# ---------------------------------------------------------------------------

def process_image_page(engine: RapidOCR, img: Image.Image) -> tuple[str, dict]:
    """
    Full pipeline: OCR → group rows → detect table → output markdown.
    Returns (markdown_string, metadata_dict).
    """
    # Step 1: OCR
    fragments = run_ocr(engine, img)

    if not fragments:
        return "(No text detected on this page)", {
            "fragments": 0, "rows": 0, "is_table": False
        }

    # Step 2: Group into rows
    rows = group_into_rows(fragments)

    # Step 3: Detect table structure
    columns = detect_column_positions(rows)
    is_table = columns is not None

    metadata = {
        "fragments": len(fragments),
        "rows": len(rows),
        "is_table": is_table,
        "columns": len(columns) if columns else 0,
    }

    # Step 4: Format output
    if is_table:
        content = rows_to_markdown_table(rows, columns)
    else:
        content = rows_to_prose(rows)

    return content, metadata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_image_pages.py <path_to_pdf>")
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

    # --- Set up output directories ---
    pages_dir = out_dir / "pages"
    images_dir = out_dir / "images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- Identify image pages ---
    image_pages = [p for p in manifest["pages"] if p["page_type"] == "image"]

    if not image_pages:
        print("No image pages to process.  All pages were text-extractable.")
        return

    print(f"Processing {len(image_pages)} image pages from {pdf_path.name}")
    print()

    # --- Initialize OCR engine ---
    print("Initializing RapidOCR…")
    engine = RapidOCR()

    # --- Open PDF ---
    doc = fitz.open(str(pdf_path))
    t0 = time.time()

    extracted = 0
    errors = 0

    for page_info in image_pages:
        page_num = page_info["page_num"]
        ref_id = page_info["reference_id"] or "unknown"

        print(f"  Page {page_num} (RefID {ref_id}):")

        try:
            # --- Rasterize ---
            print(f"    Rasterizing at {RASTER_DPI} DPI…")
            img = rasterize_page(doc, page_num, dpi=RASTER_DPI)

            # Save rasterized image for comparison
            img_path = images_dir / f"page_{page_num:03d}.png"
            img.save(str(img_path))

            # --- OCR + reconstruct ---
            print(f"    Running RapidOCR…")
            content, meta = process_image_page(engine, img)

            if not content.strip():
                print(f"    WARNING: no content extracted")

            col_info = f", {meta['columns']} columns" if meta['is_table'] else ""
            print(f"    Detected: {'table' if meta['is_table'] else 'prose'} "
                  f"({meta['fragments']} fragments, {meta['rows']} rows{col_info})")

            # --- Build page Markdown ---
            md_lines = [
                f"<!-- page: {page_num} | ref_id: {ref_id} "
                f"| has_tables: {meta['is_table']} "
                f"| extraction_method: rapidocr "
                f"| fragments: {meta['fragments']} "
                f"| needs_review: true -->",
                "",
                f"> **Note:** This page was extracted via RapidOCR with "
                f"bounding-box table reconstruction. Compare against "
                f"`images/page_{page_num:03d}.png` for verification.",
                "",
                content,
                "",
            ]

            md_text = "\n".join(md_lines)

            # --- Write ---
            page_file = pages_dir / f"page_{page_num:03d}.md"
            page_file.write_text(md_text, encoding="utf-8")
            print(f"    Saved {page_file.name} ({len(content)} chars)")
            extracted += 1

        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()
            errors += 1

    doc.close()
    elapsed = time.time() - t0

    # --- Update manifest ---
    manifest["image_extraction"] = {
        "pages_extracted": extracted,
        "errors": errors,
        "method": "rapidocr",
        "elapsed_seconds": round(elapsed, 1),
        "images_dir": str(images_dir),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"IMAGE EXTRACTION COMPLETE (RapidOCR)")
    print(f"  Pages extracted:   {extracted}")
    print(f"  Errors:            {errors}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Page markdowns:    {pages_dir}")
    print(f"  Rasterized PNGs:   {images_dir}")
    print()
    print("Next step: spot-check the .md files against the .png images,")
    print("then run assemble_documents.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
