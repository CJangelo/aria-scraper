"""
diagnostics.py — Summarize batch results and per-PDF extraction quality.

Reads batch_results.json and each PDF's manifest.json + index.json to
produce a summary table showing page counts, text vs image breakdown,
sub-document counts, and any errors.

Usage:
    uv run python diagnostics.py               # summary table
    uv run python diagnostics.py --detail       # per-PDF detail
    uv run python diagnostics.py --csv          # write diagnostics.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


OUTPUT_ROOT = Path("output")
RESULTS_FILE = Path("batch_results.json")


def load_batch_results() -> dict | None:
    """Load batch_results.json if it exists."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_manifest(stem: str) -> dict | None:
    """Load a PDF's manifest.json."""
    path = OUTPUT_ROOT / stem / "manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_index(stem: str) -> dict | None:
    """Load a PDF's final/index.json."""
    path = OUTPUT_ROOT / stem / "final" / "index.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def gather_rows(batch: dict) -> list[dict]:
    """Build one row per PDF with all diagnostic fields."""
    rows = []

    for stem, info in batch.get("pdfs", {}).items():
        row = {
            "pdf": stem,
            "status": info["status"],
            "error": info.get("error", ""),
            "failed_step": info.get("failed_step", ""),
            "elapsed_s": info.get("elapsed_seconds", 0),
            "total_pages": 0,
            "text_pages": 0,
            "image_pages": 0,
            "table_pages": 0,
            "sig_pages": 0,
            "sub_docs": 0,
            "text_extracted": 0,
            "text_errors": 0,
            "image_extracted": 0,
            "image_errors": 0,
            "assembled": 0,
        }

        manifest = load_manifest(stem)
        if manifest:
            s = manifest.get("summary", {})
            row["total_pages"] = manifest.get("total_pages", 0)
            row["text_pages"] = s.get("text_pages", 0)
            row["image_pages"] = s.get("image_pages", 0)
            row["table_pages"] = s.get("pages_with_tables", 0)
            row["sig_pages"] = s.get("signature_pages", 0)
            row["sub_docs"] = s.get("sub_document_count", 0)

            te = manifest.get("text_extraction", {})
            row["text_extracted"] = te.get("pages_extracted", 0)
            row["text_errors"] = te.get("errors", 0)

            ie = manifest.get("image_extraction", {})
            row["image_extracted"] = ie.get("pages_extracted", 0)
            row["image_errors"] = ie.get("errors", 0)

        index = load_index(stem)
        if index:
            row["assembled"] = index.get("sub_documents_assembled", 0)

        rows.append(row)

    return rows


def print_summary(rows: list[dict]):
    """Print a compact summary table."""
    if not rows:
        print("No results to display.")
        return

    # Totals
    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    fail = sum(1 for r in rows if r["status"] == "failed")
    total_pages = sum(r["total_pages"] for r in rows)
    total_text = sum(r["text_pages"] for r in rows)
    total_image = sum(r["image_pages"] for r in rows)
    total_assembled = sum(r["assembled"] for r in rows)
    total_time = sum(r["elapsed_s"] for r in rows)
    text_errors = sum(r["text_errors"] for r in rows)
    image_errors = sum(r["image_errors"] for r in rows)

    print("=" * 60)
    print("BATCH DIAGNOSTICS")
    print("=" * 60)
    print(f"  PDFs processed:    {total}  ({ok} ok, {fail} failed)")
    print(f"  Total pages:       {total_pages}")
    print(f"    Text pages:      {total_text}")
    print(f"    Image pages:     {total_image}")
    print(f"  Sub-docs assembled:{total_assembled}")
    print(f"  Extraction errors: {text_errors} text, {image_errors} image")
    print(f"  Total time:        {total_time:.0f}s ({total_time/60:.1f} min)")

    if fail > 0:
        print()
        print("FAILURES:")
        for r in rows:
            if r["status"] == "failed":
                print(f"  {r['pdf']}: {r['error']} (at {r['failed_step']})")

    print("=" * 60)


def print_detail(rows: list[dict]):
    """Print per-PDF detail as a formatted table."""
    if not rows:
        print("No results.")
        return

    # Header
    hdr = f"{'PDF':<40} {'St':>2} {'Pgs':>4} {'Txt':>4} {'Img':>4} {'Tbl':>4} {'Sub':>3} {'Asm':>3} {'Err':>3} {'Time':>6}"
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        st = "OK" if r["status"] == "ok" else "XX"
        errs = r["text_errors"] + r["image_errors"]
        name = r["pdf"][:39]
        print(
            f"{name:<40} {st:>2} {r['total_pages']:>4} {r['text_pages']:>4} "
            f"{r['image_pages']:>4} {r['table_pages']:>4} {r['sub_docs']:>3} "
            f"{r['assembled']:>3} {errs:>3} {r['elapsed_s']:>5.0f}s"
        )

    if any(r["status"] == "failed" for r in rows):
        print()
        print("FAILURES:")
        for r in rows:
            if r["status"] == "failed":
                print(f"  {r['pdf']}: {r['error']}")


def write_csv(rows: list[dict], path: str = "diagnostics.csv"):
    """Write rows to a CSV file."""
    if not rows:
        print("No data to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="ARIA batch diagnostics.")
    parser.add_argument("--detail", action="store_true", help="Per-PDF detail table.")
    parser.add_argument("--csv", action="store_true", help="Write diagnostics.csv.")
    args = parser.parse_args()

    batch = load_batch_results()
    if not batch:
        print(f"No {RESULTS_FILE} found. Run batch_run.py first.")
        sys.exit(1)

    rows = gather_rows(batch)

    if args.csv:
        write_csv(rows)
    elif args.detail:
        print_detail(rows)
    else:
        print_summary(rows)


if __name__ == "__main__":
    main()
