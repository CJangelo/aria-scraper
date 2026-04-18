"""
chunk_documents.py — Split sub-documents into overlapping text chunks
with metadata, ready for embedding and storage in ChromaDB.

Reads every final/*.md file referenced in master_index.json, parses YAML
front matter into metadata, splits the body into chunks of ~500 tokens
(~375 words) with ~50-word overlap, and writes chunks.jsonl.

Usage:
    uv run python chunk_documents.py

Output:
    chunks.jsonl  — one JSON object per line:
        {
            "chunk_id": "zepbound-tirzepatide__5273992_aria_sufficiency__001",
            "text": "...",
            "word_count": 382,
            "metadata": {
                "drug_key": "zepbound-tirzepatide",
                "drug_name": "Zepbound Tirzepatide",
                "reference_id": "5273992",
                "document_type": "aria_sufficiency",
                "source_pdf": "zepbound-tirzepatide.pdf",
                "page_range": "1-45",
                ...
            }
        }
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MASTER_INDEX = Path("master_index.json")
OUTPUT_FILE = Path("chunks.jsonl")

# Target chunk size in words (~500 tokens ≈ 375 words).
# Chunks may be slightly larger to avoid splitting mid-sentence.
TARGET_WORDS = 375

# Overlap in words between consecutive chunks (~50 tokens ≈ 50 words)
OVERLAP_WORDS = 50

# Minimum chunk size — don't create tiny fragments
MIN_WORDS = 30


# ---------------------------------------------------------------------------
# YAML front matter parsing
# ---------------------------------------------------------------------------

def parse_front_matter(text: str) -> tuple[dict, str]:
    """
    Split a markdown file into YAML front matter (as a dict) and body text.
    Returns (metadata_dict, body_string).

    Front matter is between the first pair of '---' lines.
    """
    if not text.startswith("---"):
        return {}, text

    # Find the closing ---
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    yaml_block = text[3:end].strip()
    body = text[end + 4:].strip()

    # Simple YAML parsing (our front matter is flat key: "value" pairs)
    metadata = {}
    for line in yaml_block.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"')
        metadata[key] = value

    return metadata, body


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """Split text on double newlines, filtering out empty chunks."""
    paragraphs = re.split(r"\n\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter.  Splits on period/question mark/exclamation
    followed by whitespace.  Not perfect, but good enough for FDA prose.
    Keeps the delimiter attached to the preceding sentence.
    """
    # Split but keep the delimiter
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def chunk_text(text: str, target_words: int = TARGET_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping chunks of approximately target_words.

    Strategy:
      1. Split into paragraphs.
      2. Accumulate paragraphs until the chunk exceeds target_words.
      3. If a single paragraph exceeds target_words, split it by sentences.
      4. Overlap: the last ~overlap_words of each chunk become the start
         of the next chunk.
    """
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    # First pass: break oversized paragraphs into sentence groups
    blocks = []
    for para in paragraphs:
        if word_count(para) <= target_words:
            blocks.append(para)
        else:
            # Split this paragraph by sentences and re-group
            sentences = split_into_sentences(para)
            current = []
            current_wc = 0
            for sent in sentences:
                swc = word_count(sent)
                if current_wc + swc > target_words and current:
                    blocks.append(" ".join(current))
                    current = []
                    current_wc = 0
                current.append(sent)
                current_wc += swc
            if current:
                blocks.append(" ".join(current))

    # Second pass: accumulate blocks into chunks with overlap
    chunks = []
    current_blocks = []
    current_wc = 0

    for block in blocks:
        bwc = word_count(block)

        if current_wc + bwc > target_words and current_blocks:
            # Emit current chunk
            chunk_text_str = "\n\n".join(current_blocks)
            chunks.append(chunk_text_str)

            # Build overlap: take words from the end of the current chunk
            overlap_text = " ".join(chunk_text_str.split()[-overlap_words:])
            current_blocks = [overlap_text]
            current_wc = word_count(overlap_text)

        current_blocks.append(block)
        current_wc += bwc

    # Don't forget the last chunk
    if current_blocks:
        last_chunk = "\n\n".join(current_blocks)
        # Only add if it has enough content (avoid tiny tail chunks)
        if word_count(last_chunk) >= MIN_WORDS:
            chunks.append(last_chunk)
        elif chunks:
            # Append to previous chunk instead
            chunks[-1] = chunks[-1] + "\n\n" + last_chunk

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if not MASTER_INDEX.exists():
        print(f"Error: {MASTER_INDEX} not found. Run build_drug_index.py first.")
        sys.exit(1)

    with open(MASTER_INDEX, "r", encoding="utf-8") as f:
        master = json.load(f)

    print(f"Chunking documents for {len(master)} drugs…")

    total_chunks = 0
    total_docs = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for drug_key, drug_info in master.items():
            drug_name = drug_info["drug_name"]

            for doc in drug_info["sub_documents"]:
                doc_path = Path(doc["path"])

                if not doc_path.exists():
                    print(f"  WARNING: {doc_path} not found, skipping")
                    skipped += 1
                    continue

                raw = doc_path.read_text(encoding="utf-8")
                file_meta, body = parse_front_matter(raw)

                if not body.strip():
                    skipped += 1
                    continue

                # Build metadata for this document's chunks
                chunk_meta = {
                    "drug_key": drug_key,
                    "drug_name": drug_name,
                    "reference_id": doc.get("reference_id", ""),
                    "document_type": doc.get("document_type", ""),
                    "source_pdf": doc.get("source_pdf", ""),
                    "page_range": doc.get("page_range", ""),
                    "filename": doc.get("filename", ""),
                }

                # Add any extra fields from YAML front matter
                for key in ["date", "reviewers", "subject",
                            "application_number", "applicant"]:
                    if key in file_meta:
                        chunk_meta[key] = file_meta[key]

                # Chunk the body
                chunks = chunk_text(body)

                for i, chunk in enumerate(chunks):
                    chunk_id = (
                        f"{drug_key}__{doc.get('reference_id', 'unknown')}"
                        f"__{doc.get('document_type', 'unknown')}"
                        f"__{i + 1:03d}"
                    )

                    record = {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "word_count": word_count(chunk),
                        "metadata": chunk_meta,
                    }

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

                total_docs += 1

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"CHUNKING COMPLETE")
    print(f"  Documents processed: {total_docs}")
    print(f"  Documents skipped:   {skipped}")
    print(f"  Total chunks:        {total_chunks}")
    print(f"  Output:              {OUTPUT_FILE}")
    print(f"  Target chunk size:   ~{TARGET_WORDS} words (~500 tokens)")
    print(f"  Overlap:             ~{OVERLAP_WORDS} words")
    print()

    # Quick stats on chunk sizes
    sizes = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sizes.append(rec["word_count"])

    if sizes:
        avg = sum(sizes) / len(sizes)
        print(f"  Chunk size stats:")
        print(f"    Min:     {min(sizes)} words")
        print(f"    Max:     {max(sizes)} words")
        print(f"    Mean:    {avg:.0f} words")
        print(f"    Median:  {sorted(sizes)[len(sizes)//2]} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
