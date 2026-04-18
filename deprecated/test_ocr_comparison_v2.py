"""
test_ocr_comparison_v2.py

Round 2: test EasyOCR, TrOCR, img2table, and unstructured on a single
rasterized PDF page.

Usage:
    uv run python test_ocr_comparison_v2.py <path_to_pdf> <page_number>
    uv run python test_ocr_comparison_v2.py zepbound-tirzepatide.pdf 130

Output:
    output/ocr_comparison_v2/
        page_130_original.png
        page_130_easyocr.md
        page_130_trocr.md
        page_130_img2table.md
        page_130_unstructured.md
        page_130_summary.txt

Prerequisites (install whichever you want to test):
    uv add easyocr
    uv add transformers torch        (for TrOCR)
    uv add img2table
    uv add unstructured
"""

import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output") / "ocr_comparison_v2"
RASTER_DPI = 300


# ---------------------------------------------------------------------------
# Rasterize
# ---------------------------------------------------------------------------

def rasterize_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Rasterize a single page (1-indexed) at 300 DPI."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    zoom = RASTER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------

def run_easyocr(img: Image.Image) -> tuple[str, float]:
    """Run EasyOCR — PyTorch-based, often better than Tesseract on noisy images."""
    try:
        import easyocr
        import numpy as np
    except ImportError:
        return "SKIPPED: easyocr not installed. Run: uv add easyocr", 0

    print("  Loading EasyOCR (first run downloads models)…")
    t0 = time.time()

    try:
        reader = easyocr.Reader(["en"], gpu=False)

        print("  Running EasyOCR…")
        img_array = np.array(img)
        results = reader.readtext(img_array)

        # Sort by vertical position (top of bounding box)
        results_sorted = sorted(results, key=lambda r: r[0][0][1])

        lines = []
        for bbox, text, confidence in results_sorted:
            lines.append(f"{text}  (conf: {confidence:.2f})")

        elapsed = time.time() - t0
        return "\n".join(lines), elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


def run_trocr(img: Image.Image) -> tuple[str, float]:
    """
    Run Microsoft TrOCR — transformer-based OCR for printed text.
    Uses the base model from HuggingFace.

    TrOCR works on single text lines, so we need to either:
    - Feed the whole image (gets a single line of output), or
    - Split into lines first (needs a text detector)

    We'll try the whole-image approach first, then fall back to
    a tiled approach if the output is too short.
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        return "SKIPPED: transformers/torch not installed. Run: uv add transformers torch", 0

    print("  Loading TrOCR model (first run downloads from HuggingFace)…")
    t0 = time.time()

    try:
        model_name = "microsoft/trocr-base-printed"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        print("  Running TrOCR…")

        # TrOCR works best on cropped text lines, not full pages.
        # Strategy: tile the image into horizontal strips and OCR each one.
        img_width, img_height = img.size
        strip_height = 80  # roughly one text line at 300 DPI
        overlap = 20

        lines = []
        y = 0
        while y < img_height:
            y_end = min(y + strip_height, img_height)
            strip = img.crop((0, y, img_width, y_end))

            # Skip nearly-blank strips
            import numpy as np
            strip_array = np.array(strip.convert("L"))
            if strip_array.mean() > 250:  # almost all white
                y += strip_height - overlap
                continue

            pixel_values = processor(images=strip, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_new_tokens=200)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if text.strip():
                lines.append(text.strip())

            y += strip_height - overlap

        elapsed = time.time() - t0
        return "\n".join(lines), elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


def run_img2table(img: Image.Image, pdf_path: Path, page_num: int) -> tuple[str, float]:
    """
    Run img2table — purpose-built for extracting tables from images.
    Uses OpenCV for cell detection + OCR for cell content.
    """
    try:
        from img2table.document import Image as Img2TableImage
        from img2table.ocr import TesseractOCR
    except ImportError:
        return "SKIPPED: img2table not installed. Run: uv add img2table", 0

    print("  Running img2table…")
    t0 = time.time()

    try:
        # Save image to temp file (img2table needs a file path)
        temp_img = OUTPUT_DIR / "_temp_img2table.png"
        img.save(str(temp_img))

        # Initialize OCR engine (uses Tesseract under the hood)
        ocr = TesseractOCR(lang="eng")

        # Create img2table document
        doc = Img2TableImage(src=str(temp_img))

        # Extract tables with OCR
        tables = doc.extract_tables(ocr=ocr, borderless_tables=True)

        # Convert tables to markdown
        lines = []
        for i, table in enumerate(tables):
            lines.append(f"### Table {i + 1}")
            lines.append("")

            # table.df gives a pandas DataFrame
            df = table.df
            if df is not None and not df.empty:
                lines.append(df.to_markdown(index=False))
            else:
                lines.append("(empty table detected)")
            lines.append("")

        # Clean up temp file
        try:
            temp_img.unlink()
        except Exception:
            pass

        elapsed = time.time() - t0

        if not lines:
            return "(no tables detected)", elapsed

        return "\n".join(lines), elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


def run_unstructured(pdf_path: Path, page_num: int) -> tuple[str, float]:
    """
    Run unstructured — high-level document parsing that chains
    multiple tools together.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return "SKIPPED: unstructured not installed. Run: uv add unstructured", 0

    print("  Running unstructured…")
    t0 = time.time()

    try:
        # Extract just the target page
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            include_page_breaks=True,
            pages=[page_num],
        )

        lines = []
        for element in elements:
            element_type = type(element).__name__
            text = str(element)
            if text.strip():
                lines.append(f"[{element_type}] {text}")

        elapsed = time.time() - t0
        return "\n".join(lines), elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_ocr_comparison_v2.py <path_to_pdf> <page_number>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    page_num = int(sys.argv[2])

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"page_{page_num}"

    # --- Rasterize ---
    print(f"Rasterizing page {page_num} at {RASTER_DPI} DPI…")
    img = rasterize_page(pdf_path, page_num)
    img_path = OUTPUT_DIR / f"{prefix}_original.png"
    img.save(str(img_path))
    print(f"  Saved {img_path}")
    print()

    # --- Run each tool ---
    results = {}

    print("[1/4] EASYOCR")
    text, elapsed = run_easyocr(img)
    results["easyocr"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    print("[2/4] TROCR")
    text, elapsed = run_trocr(img)
    results["trocr"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    print("[3/4] IMG2TABLE")
    text, elapsed = run_img2table(img, pdf_path, page_num)
    results["img2table"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    print("[4/4] UNSTRUCTURED")
    text, elapsed = run_unstructured(pdf_path, page_num)
    results["unstructured"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    # --- Save results ---
    for tool_name, result in results.items():
        md_path = OUTPUT_DIR / f"{prefix}_{tool_name}.md"
        md_path.write_text(result["text"], encoding="utf-8")
        print(f"  Saved {md_path.name}")

    # --- Summary ---
    summary_lines = [
        f"OCR Comparison v2 — Page {page_num} of {pdf_path.name}",
        f"Rasterized at {RASTER_DPI} DPI",
        "",
        f"{'Tool':<15} {'Chars':>8} {'Time':>8}  Status",
        "-" * 50,
    ]

    for tool_name, result in results.items():
        text = result["text"]
        elapsed = result["elapsed"]
        if text.startswith("SKIPPED"):
            status = "not installed"
        elif text.startswith("ERROR"):
            status = "error"
        else:
            status = "ok"
        summary_lines.append(
            f"{tool_name:<15} {len(text):>8} {elapsed:>7.1f}s  {status}"
        )

    summary_lines.extend([
        "",
        "Compare each .md file against the original .png to pick a winner.",
        f"Files are in: {OUTPUT_DIR.resolve()}",
    ])

    summary = "\n".join(summary_lines)
    summary_path = OUTPUT_DIR / f"{prefix}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
