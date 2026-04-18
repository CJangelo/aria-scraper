"""Embed chunks.jsonl → ChromaDB with metadata. Supports incremental updates."""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_PATH = Path("chunks.jsonl")
DB_PATH = Path("chroma_db")
BATCH_SIZE = 500

METADATA_FIELDS = ["drug_key", "drug_name", "document_type", "reference_id", "source_pdf", "page_range", "filename"]


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    # 1. Load chunks
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH}")

    # 2. Load embedding model (downloads ~80MB first time)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Set up ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="aria_memos",
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Diff against existing collection
    existing_ids = set(collection.get(include=[])["ids"])
    jsonl_ids = {c["chunk_id"] for c in chunks}

    to_add = [c for c in chunks if c["chunk_id"] not in existing_ids]
    to_remove = existing_ids - jsonl_ids

    print(f"Collection has {len(existing_ids):,} existing chunks")
    print(f"  {len(to_add):,} new chunks to embed")
    print(f"  {len(to_remove):,} stale chunks to remove")

    # 5. Delete stale chunks
    if to_remove:
        remove_list = list(to_remove)
        for i in range(0, len(remove_list), BATCH_SIZE):
            collection.delete(ids=remove_list[i : i + BATCH_SIZE])
        print(f"Removed {len(to_remove):,} stale chunks")

    # 6. Embed and upsert new chunks
    if not to_add:
        print("Nothing new to embed.")
    else:
        for i in range(0, len(to_add), BATCH_SIZE):
            batch = to_add[i : i + BATCH_SIZE]

            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {field: c.get("metadata", {}).get(field, "") for field in METADATA_FIELDS}
                for c in batch
            ]

            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  {min(i + BATCH_SIZE, len(to_add)):,} / {len(to_add):,} embedded")

    print(f"\nDone. Collection has {collection.count():,} documents.")
    print(f"DB stored at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
