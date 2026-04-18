"""
Script 4 of 4 — assemble_documents.py

Read the manifest and all per-page markdown files from Scripts 2 and 3,
group by sub-document (Reference ID), clean header/footer noise, add
YAML front matter metadata, and write one final markdown file per
sub-document.

Usage:
    uv run python assemble_documents.py <path_to_pdf>
    uv run python assemble_documents.py zepbound-tirzepatide.pdf

Reads:
    output/<pdf_stem>/manifest.json
    output/<pdf_stem>/pages/page_*.md

Writes:
    output/<pdf_stem>/final/<ref_id>_<doc_type>.md
    output/<pdf_stem>/final/index.json
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("output")


# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

FOOTER_PATTERNS = [
    re.compile(r"^Reference ID:\s*\d+\s*$", re.MULTILINE),
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.MULTILINE),
    re.compile(r"^\d+\s+\d+$", re.MULTILINE),
    re.compile(r"^ReferenceID:\s*\d+\s*$", re.MULTILINE),
]

HEADER_PATTERNS = [
    re.compile(
        r"^U\.?S\.?\s*FOOD\s*&?\s*DRUG\s*$", re.MULTILINE | re.IGNORECASE
    ),
    re.compile(
        r"^ADMINISTRATION\s*$", re.MULTILINE | re.IGNORECASE
    ),
    re.compile(
        r"^U\.S\.\s*Food\s*and\s*Drug\s*Administration\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^NDA\s+\d+\s*[-–]\s*\w+\s+\w+\s+\w+\s*$", re.MULTILINE
    ),
]


def strip_noise(text: str) -> str:
    """Remove header/footer noise from a page's text."""
    for pattern in FOOTER_PATTERNS:
        text = pattern.sub("", text)
    for pattern in HEADER_PATTERNS:
        text = pattern.sub("", text)

    text = re.sub(r"^<!--.*?-->\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^>\s*\*\*Note:\*\*.*?$", "", text, flags=re.MULTILINE
    )
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata_from_first_page(text: str) -> dict:
    """
    Try to extract structured metadata from the first page of a
    sub-document.
    """
    meta = {}

    m = re.search(r"Date:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        meta["date"] = m.group(1).strip()

    m = re.search(
        r"Reviewer\(?s?\)?:\s*(.+?)(?:\n\n|\nTeam|\nDivision|\nSubject)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        reviewers = m.group(1).strip()
        reviewers = re.sub(r"\n\s*", ", ", reviewers)
        meta["reviewers"] = reviewers

    m = re.search(
        r"Subject:\s*(.+?)(?:\nDrug Name|\nApplication|\n\n)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        subject = m.group(1).strip()
        subject = re.sub(r"\n\s*", " ", subject)
        meta["subject"] = subject

    m = re.search(
        r"Drug Name\(?s?\)?:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    if m:
        meta["drug_name"] = m.group(1).strip()

    m = re.search(
        r"Application Type/?Number:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    if m:
        meta["application_number"] = m.group(1).strip()

    m = re.search(
        r"Applic?ant(?:/[Ss]ponsor)?(?:\s*Name)?:\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if m:
        meta["applicant"] = m.group(1).strip()

    return meta


# ---------------------------------------------------------------------------
# YAML front matter
# ---------------------------------------------------------------------------

def build_front_matter(ref_id: str, doc_type: str | None,
                       page_range: str, page_count: int,
                       extracted_meta: dict,
                       source_file: str) -> str:
    """Build a YAML front matter block for a sub-document."""
    lines = ["---"]
    lines.append(f"reference_id: \"{ref_id}\"")
    lines.append(f"document_type: \"{doc_type or 'unknown'}\"")
    lines.append(f"source_file: \"{source_file}\"")
    lines.append(f"page_range: \"{page_range}\"")
    lines.append(f"page_count: {page_count}")

    for key, value in extracted_meta.items():
        safe_value = str(value).replace('"', '\\"')
        lines.append(f"{key}: \"{safe_value}\"")

    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# run() — importable entry point for batch processing
# ---------------------------------------------------------------------------

def run(pdf_path: Path) -> dict:
    """
    Assemble final sub-documents from per-page markdown.  Returns:
        {"status": "ok", "assembled": int, "skipped_pages": [...]}
    Raises on fatal errors.
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
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Identify pages to skip
    skip_pages = set()
    for p in manifest["pages"]:
        if p["is_signature_page"]:
            skip_pages.add(p["page_num"])
        if p["is_cover_page"]:
            skip_pages.add(p["page_num"])

    print(f"Assembling sub-documents from {pdf_path.name}")
    print(f"  Skipping {len(skip_pages)} signature/cover pages: {sorted(skip_pages)}")

    sub_docs = manifest["sub_documents"]
    index_entries = []
    assembled = 0

    for ref_id, sd_info in sub_docs.items():
        doc_type = sd_info.get("document_type_hint") or "unknown"
        page_nums = [
            p["page_num"] for p in manifest["pages"]
            if p["reference_id"] == ref_id
            and p["page_num"] not in skip_pages
        ]

        if not page_nums:
            print(f"  RefID {ref_id} [{doc_type}]: no usable pages, skipping")
            continue

        print(f"  RefID {ref_id} [{doc_type}]: pages {page_nums[0]}–{page_nums[-1]} "
              f"({len(page_nums)} pages)")

        page_contents = []
        first_page_raw = None

        for page_num in sorted(page_nums):
            page_file = pages_dir / f"page_{page_num:03d}.md"

            if not page_file.exists():
                print(f"    WARNING: {page_file.name} not found, skipping")
                continue

            raw = page_file.read_text(encoding="utf-8")
            cleaned = strip_noise(raw)

            if not cleaned.strip():
                continue

            if first_page_raw is None:
                first_page_raw = raw

            page_contents.append({
                "page_num": page_num,
                "content": cleaned,
            })

        if not page_contents:
            print(f"    WARNING: no content after cleaning, skipping")
            continue

        extracted_meta = {}
        if first_page_raw:
            extracted_meta = extract_metadata_from_first_page(first_page_raw)

        page_range = f"{page_nums[0]}-{page_nums[-1]}"
        front_matter = build_front_matter(
            ref_id=ref_id,
            doc_type=doc_type,
            page_range=page_range,
            page_count=len(page_contents),
            extracted_meta=extracted_meta,
            source_file=pdf_path.name,
        )

        body_parts = []
        for pc in page_contents:
            body_parts.append(f"<!-- page {pc['page_num']} -->")
            body_parts.append(pc["content"])

        body = "\n\n".join(body_parts)
        full_doc = front_matter + "\n\n" + body + "\n"

        filename = f"{ref_id}_{doc_type}.md"
        out_path = final_dir / filename
        out_path.write_text(full_doc, encoding="utf-8")
        print(f"    → {filename} ({len(full_doc)} chars)")

        index_entries.append({
            "reference_id": ref_id,
            "document_type": doc_type,
            "filename": filename,
            "page_range": page_range,
            "page_count": len(page_contents),
            "char_count": len(full_doc),
            "metadata": extracted_meta,
        })

        assembled += 1

    # Write index
    index = {
        "source_file": pdf_path.name,
        "sub_documents_assembled": assembled,
        "pages_skipped": sorted(skip_pages),
        "documents": index_entries,
    }

    index_path = final_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "assembled": assembled,
        "skipped_pages": sorted(skip_pages),
        "index_path": str(index_path),
        "documents": index_entries,
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python assemble_documents.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    try:
        result = run(pdf_path)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"ASSEMBLY COMPLETE")
    print(f"  Sub-documents assembled: {result['assembled']}")
    print(f"  Pages skipped:           {len(result['skipped_pages'])}")
    print(f"  Index:                   {result['index_path']}")
    print()
    print("Final markdown files:")
    for entry in result["documents"]:
        meta_summary = ""
        if "drug_name" in entry["metadata"]:
            meta_summary = f" — {entry['metadata']['drug_name']}"
        print(f"  {entry['filename']}{meta_summary}")
    print("=" * 60)


if __name__ == "__main__":
    main()
