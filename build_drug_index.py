"""
build_drug_index.py — Build a master index grouping all sub-documents by drug.

Reads every output/*/final/index.json produced by the pipeline and groups
sub-documents by drug, using the PDF stem (e.g. "zepbound-tirzepatide") as
the grouping key.  This is more reliable than the metadata drug_name field,
which only appears on certain sub-document types.

Writes master_index.json to the project root.

Usage:
    uv run python build_drug_index.py
"""

import json
from collections import defaultdict
from pathlib import Path


OUTPUT_ROOT = Path("output")
MASTER_INDEX_FILE = Path("master_index.json")


def drug_name_from_stem(stem: str) -> str:
    """
    Convert a PDF stem like 'zepbound-tirzepatide' into a readable drug name.
    Replaces hyphens with spaces and title-cases.
    Strips trailing '-0', '-1' suffixes from duplicate downloads.
    """
    clean = stem
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        clean = parts[0]

    return clean.replace("-", " ").title()


def build_index() -> dict:
    """
    Scan all index.json files and group sub-documents by drug (PDF stem).
    Returns a dict keyed by cleaned PDF stem.
    """
    drugs = defaultdict(lambda: {
        "drug_name": None,
        "stems": set(),
        "application_numbers": set(),
        "source_pdfs": set(),
        "sub_documents": [],
    })

    index_files = sorted(OUTPUT_ROOT.glob("*/final/index.json"))

    if not index_files:
        print("No index.json files found. Run the pipeline first.")
        return {}

    print(f"Reading {len(index_files)} index files…")

    for index_path in index_files:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        source_pdf = index.get("source_file", "unknown")
        pdf_stem = index_path.parent.parent.name  # output/<stem>/final/

        # Group by cleaned stem (merges "arakoda-tafenoquine" and "arakoda-tafenoquine-0")
        parts = pdf_stem.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            group_key = parts[0]
        else:
            group_key = pdf_stem

        entry = drugs[group_key]
        entry["drug_name"] = drug_name_from_stem(group_key)
        entry["stems"].add(pdf_stem)
        entry["source_pdfs"].add(source_pdf)

        for doc in index.get("documents", []):
            meta = doc.get("metadata", {})

            if meta.get("application_number"):
                entry["application_numbers"].add(meta["application_number"])

            entry["sub_documents"].append({
                "reference_id": doc.get("reference_id"),
                "document_type": doc.get("document_type"),
                "filename": doc.get("filename"),
                "page_range": doc.get("page_range"),
                "page_count": doc.get("page_count"),
                "char_count": doc.get("char_count"),
                "source_pdf": source_pdf,
                "metadata": meta,
                "path": str(
                    OUTPUT_ROOT / pdf_stem / "final" / doc.get("filename", "")
                ),
            })

    # Convert sets to sorted lists for JSON serialization
    result = {}
    for key in sorted(drugs.keys()):
        entry = drugs[key]
        result[key] = {
            "drug_name": entry["drug_name"],
            "stems": sorted(entry["stems"]),
            "application_numbers": sorted(entry["application_numbers"]),
            "source_pdfs": sorted(entry["source_pdfs"]),
            "sub_document_count": len(entry["sub_documents"]),
            "total_chars": sum(
                d.get("char_count", 0) for d in entry["sub_documents"]
            ),
            "sub_documents": entry["sub_documents"],
        }

    return result


def main():
    index = build_index()

    if not index:
        return

    with open(MASTER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    # Summary
    total_docs = sum(v["sub_document_count"] for v in index.values())
    total_chars = sum(v["total_chars"] for v in index.values())

    # Count how many drugs merged from multiple stems
    merged = sum(1 for v in index.values() if len(v["stems"]) > 1)

    print()
    print("=" * 60)
    print(f"MASTER INDEX: {MASTER_INDEX_FILE}")
    print(f"  Drugs:           {len(index)}")
    print(f"  Sub-documents:   {total_docs}")
    print(f"  Total chars:     {total_chars:,}")
    if merged:
        print(f"  Merged stems:    {merged} drugs had multiple PDFs")
    print()

    # Top 10 by sub-document count
    by_count = sorted(index.items(), key=lambda x: x[1]["sub_document_count"], reverse=True)
    print("Top 10 drugs by sub-document count:")
    for key, val in by_count[:10]:
        print(f"  {val['drug_name']}: {val['sub_document_count']} docs, {val['total_chars']:,} chars")

    print("=" * 60)


if __name__ == "__main__":
    main()
