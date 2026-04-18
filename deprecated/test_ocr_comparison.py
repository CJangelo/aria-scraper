"""
test_ocr_comparison.py

Run multiple OCR/table extraction tools on a single rasterized PDF page
and save each result for side-by-side comparison.

This is a diagnostic script — not part of the main pipeline.  Use it to
pick the best tool for image-based pages, then plug that tool into
extract_image_pages.py.

Usage:
    uv run python test_ocr_comparison.py <path_to_pdf> <page_number>
    uv run python test_ocr_comparison.py zepbound-tirzepatide.pdf 130

Output:
    output/ocr_comparison/
        page_130_original.png        (rasterized page for reference)
        page_130_surya.md            (surya result)
        page_130_marker.md           (marker-pdf result)
        page_130_paddleocr.md        (PaddleOCR result)
        page_130_summary.txt         (side-by-side char counts + notes)

Prerequisites (install whichever you want to test):
    uv add surya-ocr
    uv add marker-pdf
    uv add paddleocr paddlepaddle
"""

import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output") / "ocr_comparison"
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
# Tool runners — each returns (markdown_string, elapsed_seconds)
# If the tool isn't installed, it returns a helpful error message.
# ---------------------------------------------------------------------------

def run_surya(img: Image.Image, pdf_path: Path, page_num: int) -> tuple[str, float]:
    """Run surya OCR + table recognition."""
    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError:
        return "SKIPPED: surya not installed. Run: uv add surya-ocr", 0

    print("  Loading surya models…")
    t0 = time.time()

    try:
        det_model = load_det_model()
        det_processor = load_det_processor()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()

        print("  Running surya OCR…")
        results = run_ocr(
            [img],
            [["en"]],
            det_model,
            det_processor,
            rec_model,
            rec_processor,
        )

        # Extract text lines sorted by vertical position
        lines = []
        if results and len(results) > 0:
            for text_line in sorted(results[0].text_lines, key=lambda l: l.bbox[1]):
                lines.append(text_line.text)

        elapsed = time.time() - t0
        return "\n".join(lines), elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


def run_marker(pdf_path: Path, page_num: int) -> tuple[str, float]:
    """Run marker-pdf on a single page."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
    except ImportError:
        # Try alternate import paths (marker API has changed across versions)
        try:
            from marker.convert import convert_single_pdf
        except ImportError:
            return "SKIPPED: marker-pdf not installed. Run: uv add marker-pdf", 0

        # Use the simpler legacy API
        print("  Running marker-pdf (legacy API)…")
        t0 = time.time()
        try:
            text, images, metadata = convert_single_pdf(
                str(pdf_path),
                max_pages=1,
                start_page=page_num - 1,
            )
            elapsed = time.time() - t0
            return text, elapsed
        except Exception as exc:
            elapsed = time.time() - t0
            return f"ERROR: {exc}", elapsed

    print("  Loading marker-pdf models…")
    t0 = time.time()

    try:
        config_parser = ConfigParser({
            "page_range": f"{page_num - 1}-{page_num - 1}",
        })
        models = create_model_dict()
        converter = PdfConverter(
            artifact_dict=models,
            config=config_parser.generate_config_dict(),
        )

        print("  Running marker-pdf conversion…")
        result = converter(str(pdf_path))
        markdown = result.markdown

        elapsed = time.time() - t0
        return markdown, elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        return f"ERROR: {exc}", elapsed


def run_paddleocr(img: Image.Image) -> tuple[str, float]:
    """Run PaddleOCR with table structure recognition."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "SKIPPED: PaddleOCR not installed. Run: uv add paddleocr paddlepaddle", 0

    print("  Loading PaddleOCR models…")
    t0 = time.time()

    try:
        import numpy as np

        # Initialize with table structure recognition enabled
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )

        print("  Running PaddleOCR…")
        img_array = np.array(img)
        results = ocr.ocr(img_array, cls=True)

        # Extract text sorted by vertical position
        lines = []
        if results and results[0]:
            # Sort by y-coordinate of the top-left corner
            sorted_results = sorted(results[0], key=lambda r: r[0][0][1])
            for detection in sorted_results:
                bbox, (text, confidence) = detection
                lines.append(f"{text}  (conf: {confidence:.2f})")

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
        print("Usage: python test_ocr_comparison.py <path_to_pdf> <page_number>")
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

    print("[1/3] SURYA")
    text, elapsed = run_surya(img, pdf_path, page_num)
    results["surya"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    print("[2/3] MARKER-PDF")
    text, elapsed = run_marker(pdf_path, page_num)
    results["marker"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    print("[3/3] PADDLEOCR")
    text, elapsed = run_paddleocr(img)
    results["paddleocr"] = {"text": text, "elapsed": elapsed}
    print(f"  → {len(text)} chars, {elapsed:.1f}s")
    print()

    # --- Save results ---
    for tool_name, result in results.items():
        md_path = OUTPUT_DIR / f"{prefix}_{tool_name}.md"
        md_path.write_text(result["text"], encoding="utf-8")
        print(f"  Saved {md_path.name}")

    # --- Summary ---
    summary_lines = [
        f"OCR Comparison — Page {page_num} of {pdf_path.name}",
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
