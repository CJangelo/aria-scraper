"""
batch_run.py — Run the 4-step ARIA memo pipeline on all PDFs in aria_memos/.

For each PDF:
  1. classify_pages.run()   → manifest.json
  2. extract_text_pages.run() → pages/*.md (text pages)
  3. extract_image_pages.run() → pages/*.md + images/*.png (OCR pages)
  4. assemble_documents.run() → final/*.md + final/index.json

If any step fails for a PDF, the error is logged and the batch continues
with the next PDF.  Results are written to batch_results.json.

Usage:
    uv run python batch_run.py                  # process all PDFs
    uv run python batch_run.py --dry-run        # list PDFs without processing
    uv run python batch_run.py --only new       # skip PDFs that already have output/
    uv run python batch_run.py --only failed    # re-run only previously failed PDFs
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# These imports work because all scripts live in the same directory.
# Each script's run() accepts a Path and returns a dict.
from classify_pages import run as classify
from extract_text_pages import run as extract_text
from extract_image_pages import run as extract_images
from assemble_documents import run as assemble


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMOS_DIR = Path("aria_memos")
OUTPUT_ROOT = Path("output")
RESULTS_FILE = Path("batch_results.json")

# The 4 pipeline steps in order.  Each is (name, function).
STEPS = [
    ("classify", classify),
    ("extract_text", extract_text),
    ("extract_images", extract_images),
    ("assemble", assemble),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pdfs() -> list[Path]:
    """Return all .pdf files in aria_memos/, sorted alphabetically."""
    if not MEMOS_DIR.exists():
        print(f"Error: {MEMOS_DIR}/ directory not found.")
        sys.exit(1)
    pdfs = sorted(MEMOS_DIR.glob("*.pdf"))
    return pdfs


def load_previous_results() -> dict:
    """Load batch_results.json if it exists, else return empty dict."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    """Write batch_results.json."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the ARIA memo extraction pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List PDFs that would be processed, but don't run anything.",
    )
    parser.add_argument(
        "--only", choices=["new", "failed"],
        help=(
            "'new' = skip PDFs that already have an output/ folder. "
            "'failed' = only re-run PDFs that failed last time."
        ),
    )
    args = parser.parse_args()

    all_pdfs = find_pdfs()
    if not all_pdfs:
        print(f"No PDFs found in {MEMOS_DIR}/")
        sys.exit(0)

    # --- Filter PDFs based on --only flag ---
    previous = load_previous_results()

    if args.only == "new":
        pdfs = [p for p in all_pdfs if not (OUTPUT_ROOT / p.stem).exists()]
        print(f"Found {len(all_pdfs)} PDFs, {len(pdfs)} are new (no output/ yet).")
    elif args.only == "failed":
        failed_stems = {
            stem for stem, info in previous.get("pdfs", {}).items()
            if info.get("status") == "failed"
        }
        pdfs = [p for p in all_pdfs if p.stem in failed_stems]
        print(f"Found {len(all_pdfs)} PDFs, {len(pdfs)} previously failed.")
    else:
        pdfs = all_pdfs
        print(f"Found {len(pdfs)} PDFs in {MEMOS_DIR}/")

    if not pdfs:
        print("Nothing to process.")
        sys.exit(0)

    # --- Dry run ---
    if args.dry_run:
        print()
        for p in pdfs:
            print(f"  {p.name}")
        print(f"\n{len(pdfs)} PDFs would be processed. Use without --dry-run to execute.")
        sys.exit(0)

    # --- Process ---
    results = {
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pdfs": len(pdfs),
        "pdfs": {},
    }

    succeeded = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs, 1):
        stem = pdf_path.stem
        print()
        print("=" * 60)
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        print("=" * 60)

        pdf_result = {
            "file": str(pdf_path),
            "status": "ok",
            "steps": {},
            "error": None,
        }

        t0 = time.time()

        for step_name, step_fn in STEPS:
            print(f"\n  → Step: {step_name}")
            try:
                step_result = step_fn(pdf_path)
                pdf_result["steps"][step_name] = step_result
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"  ✗ FAILED at {step_name}: {error_msg}")
                traceback.print_exc()

                pdf_result["status"] = "failed"
                pdf_result["failed_step"] = step_name
                pdf_result["error"] = error_msg
                break  # stop this PDF, move to next

        elapsed = round(time.time() - t0, 1)
        pdf_result["elapsed_seconds"] = elapsed

        if pdf_result["status"] == "ok":
            print(f"\n  ✓ Done in {elapsed}s")
            succeeded += 1
        else:
            failed += 1

        results["pdfs"][stem] = pdf_result

        # Save after every PDF so you don't lose progress on a crash
        save_results(results)

    # --- Final summary ---
    results["run_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["succeeded"] = succeeded
    results["failed"] = failed
    save_results(results)

    print()
    print("=" * 60)
    print(f"BATCH COMPLETE")
    print(f"  Total:     {len(pdfs)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Results:   {RESULTS_FILE}")

    if failed > 0:
        print()
        print("Failed PDFs:")
        for stem, info in results["pdfs"].items():
            if info["status"] == "failed":
                print(f"  {stem}: {info['error']} (at {info['failed_step']})")
        print()
        print("Re-run failures with:  uv run python batch_run.py --only failed")

    print("=" * 60)


if __name__ == "__main__":
    main()
