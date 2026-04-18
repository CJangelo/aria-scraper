"""
Script 1 of 4 — classify_pages.py

Scan every page of an ARIA memo PDF and produce a manifest.json that
classifies each page as text-extractable or image-based, identifies the
sub-document it belongs to (via Reference ID), and flags whether the page
contains tables.

Usage:
    uv run python classify_pages.py <path_to_pdf>
    uv run python classify_pages.py aria_memos/zepbound-tirzepatide.pdf

Output:
    output/<pdf_stem>/manifest.json

The manifest is the input for Scripts 2, 3, and 4.
"""

import json
import re
import sys
import time
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pages with fewer extracted characters than this are classified as "image"
TEXT_CHAR_THRESHOLD = 50

# Output root
OUTPUT_ROOT = Path("output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_reference_id(text: str) -> str | None:
    """
    Pull the Reference ID from a page's text.  These appear as footer
    lines like 'Reference ID: 5273992'.  We want the *footer* one, not
    inline citations like 'DARRTS Reference ID: ...'.
    """
    # Find all 'Reference ID: NNNNNNN' patterns
    matches = re.findall(r"(?<!DARRTS )Reference ID:\s*(\d+)", text)
    if matches:
        # The footer reference ID is typically the last one on the page
        return matches[-1]
    return None


def detect_page_type(page_text: str) -> str:
    """Classify a page based on how much extractable text it has."""
    # Strip whitespace and common boilerplate
    cleaned = page_text.strip()
    # Remove header/footer noise (Reference ID lines, page numbers)
    cleaned = re.sub(r"Reference ID:\s*\d+", "", cleaned)
    cleaned = re.sub(r"Page\s+\d+\s+of\s+\d+", "", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) < TEXT_CHAR_THRESHOLD:
        return "image"
    return "text"


def detect_tables_pymupdf(page: fitz.Page) -> int:
    """Use PyMuPDF's built-in table finder to count tables on a page."""
    try:
        tables = page.find_tables()
        return len(tables.tables)
    except Exception:
        return 0


def detect_tables_pdfplumber(page) -> int:
    """Use pdfplumber's table detection as a second opinion."""
    try:
        tables = page.find_tables()
        return len(tables)
    except Exception:
        return 0


def detect_document_type(text: str) -> str | None:
    """
    Try to identify what kind of FDA review document this page belongs to
    based on header keywords.
    """
    patterns = [
        (r"ARIA Sufficiency", "aria_sufficiency"),
        (r"Expedited ARIA Sufficiency", "aria_sufficiency_expedited"),
        (r"Clinical Inspection Summary", "clinical_inspection"),
        (r"MEMORANDUM.*REVIEW OF REVISED LABEL", "labeling_review"),
        (r"MEMORANDUM.*MEDICATION ERROR", "medication_error"),
        (r"Immunogenicity", "immunogenicity"),
        (r"Clinical Pharmacology", "clinical_pharmacology"),
        (r"Clinical Outcomes Assessment", "clinical_outcomes"),
        (r"Pregnancy.*Lactation", "pregnancy_lactation"),
        (r"Risk Evaluation and Mitigation", "rems"),
        (r"Dissolution", "dissolution"),
    ]
    for pattern, doc_type in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return doc_type
    return None


def classify_pdf(pdf_path: Path) -> dict:
    """
    Main classification function.  Returns a manifest dict with:
      - source_file: original PDF filename
      - total_pages: page count
      - sub_documents: dict keyed by Reference ID with page ranges
      - pages: list of per-page classification dicts
    """
    pdf_path = Path(pdf_path)

    # Open with both libraries
    mu_doc = fitz.open(str(pdf_path))
    pl_doc = pdfplumber.open(str(pdf_path))

    pages = []
    ref_id_sequence = []  # track Reference ID per page for sub-doc grouping

    print(f"Classifying {mu_doc.page_count} pages in {pdf_path.name}…")
    t0 = time.time()

    for i in range(mu_doc.page_count):
        mu_page = mu_doc[i]
        pl_page = pl_doc.pages[i]

        # --- text extraction ---
        text_mu = mu_page.get_text("text")
        text_pl = pl_page.extract_text() or ""

        # Use the longer extraction (they sometimes differ)
        text = text_mu if len(text_mu) >= len(text_pl) else text_pl

        # --- classification ---
        page_type = detect_page_type(text)
        ref_id = extract_reference_id(text_mu) or extract_reference_id(text_pl)
        ref_id_sequence.append(ref_id)

        # --- table detection ---
        tables_mu = detect_tables_pymupdf(mu_page)
        tables_pl = detect_tables_pdfplumber(pl_page)
        has_tables = max(tables_mu, tables_pl) > 0
        table_count = max(tables_mu, tables_pl)

        # --- document type detection (only on first occurrence) ---
        doc_type = detect_document_type(text)

        # --- detect signature pages (low value, can skip later) ---
        is_signature = bool(re.search(
            r"Signature Page|electronic record that was signed", text
        ))

        # --- detect cover/separator pages ---
        is_cover = bool(re.search(
            r"^CENTER FOR DRUG EVALUATION AND\s*RESEARCH",
            text.strip(),
            re.MULTILINE
        )) and len(text.strip()) < 300

        page_info = {
            "page_num": i + 1,           # 1-indexed for human readability
            "page_type": page_type,
            "reference_id": ref_id,
            "char_count": len(text.strip()),
            "has_tables": has_tables,
            "table_count": table_count,
            "document_type_hint": doc_type,
            "is_signature_page": is_signature,
            "is_cover_page": is_cover,
        }
        pages.append(page_info)

        # Progress indicator every 25 pages
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{mu_doc.page_count} pages classified")

    mu_doc.close()
    pl_doc.close()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # --- Build sub-document groupings ---
    sub_documents = {}
    for p in pages:
        rid = p["reference_id"]
        if rid is None:
            continue
        if rid not in sub_documents:
            sub_documents[rid] = {
                "reference_id": rid,
                "first_page": p["page_num"],
                "last_page": p["page_num"],
                "page_count": 0,
                "text_pages": 0,
                "image_pages": 0,
                "pages_with_tables": 0,
                "document_type_hint": None,
            }
        sd = sub_documents[rid]
        sd["last_page"] = p["page_num"]
        sd["page_count"] += 1
        if p["page_type"] == "text":
            sd["text_pages"] += 1
        else:
            sd["image_pages"] += 1
        if p["has_tables"]:
            sd["pages_with_tables"] += 1
        if p["document_type_hint"] and sd["document_type_hint"] is None:
            sd["document_type_hint"] = p["document_type_hint"]

    # --- Summary stats ---
    total_text = sum(1 for p in pages if p["page_type"] == "text")
    total_image = sum(1 for p in pages if p["page_type"] == "image")
    total_tables = sum(1 for p in pages if p["has_tables"])
    total_sig = sum(1 for p in pages if p["is_signature_page"])

    manifest = {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path.resolve()),
        "total_pages": len(pages),
        "summary": {
            "text_pages": total_text,
            "image_pages": total_image,
            "pages_with_tables": total_tables,
            "signature_pages": total_sig,
            "sub_document_count": len(sub_documents),
        },
        "sub_documents": sub_documents,
        "pages": pages,
    }

    return manifest


# ---------------------------------------------------------------------------
# run() — importable entry point for batch processing
# ---------------------------------------------------------------------------

def run(pdf_path: Path) -> dict:
    """
    Classify a PDF and write the manifest.  Returns a result dict:
        {"status": "ok", "manifest_path": ..., "summary": ...}
    or raises an exception on failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    manifest = classify_pdf(pdf_path)

    # Write output
    out_dir = OUTPUT_ROOT / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "total_pages": manifest["total_pages"],
        "summary": manifest["summary"],
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_pages.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    try:
        result = run(pdf_path)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # --- Print summary ---
    s = result["summary"]
    print()
    print("=" * 60)
    print(f"MANIFEST: {result['manifest_path']}")
    print(f"  Total pages:       {result['total_pages']}")
    print(f"  Text pages:        {s['text_pages']}")
    print(f"  Image pages:       {s['image_pages']}")
    print(f"  Pages w/ tables:   {s['pages_with_tables']}")
    print(f"  Signature pages:   {s['signature_pages']}")
    print(f"  Sub-documents:     {s['sub_document_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
