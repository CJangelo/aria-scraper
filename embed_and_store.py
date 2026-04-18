"""Step 3: Embed chunks.jsonl → ChromaDB with metadata."""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_PATH = Path("chunks.jsonl")
DB_PATH = Path("chroma_db")
BATCH_SIZE = 500

METADATA_FIELDS = ["drug_key", "drug_name", "document_type", "reference_id", "source_pdf"]


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    # 1. Load chunks
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks):,} chunks")

    # 2. Load embedding model (downloads ~80MB first time)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Set up ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="aria_memos",
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Batch embed + upsert
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
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
        print(f"  {i + len(batch):,} / {len(chunks):,}")

    print(f"\nDone. Collection has {collection.count():,} documents.")
    print(f"DB stored at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()