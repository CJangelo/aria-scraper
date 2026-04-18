"""Retrieve chunks from ChromaDB by semantic similarity."""
import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = Path("chroma_db")
MASTER_INDEX = Path("master_index.json")


def retrieve(query: str, n: int = 5, drug: str = None, doctype: str = None) -> list[dict]:
    """
    Query ChromaDB and return the top-n matching chunks.

    Returns a list of dicts with keys: text, metadata, similarity.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("aria_memos")

    query_embedding = model.encode(query).tolist()

    # Build metadata filter
    where = None
    conditions = []
    if drug:
        if not MASTER_INDEX.exists():
            raise FileNotFoundError("master_index.json not found. Run build_drug_index.py first.")
        with open(MASTER_INDEX, "r", encoding="utf-8") as f:
            master = json.load(f)
        matching_keys = [k for k in master if drug.lower() in k.lower()]
        if matching_keys:
            print(f"Matched drug(s): {', '.join(sorted(matching_keys))}")
            conditions.append({"drug_key": {"$in": matching_keys}})
        else:
            print(f"No drugs matched '{drug}'")
            return []
    if doctype:
        conditions.append({"document_type": doctype})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity": 1 - dist,
        })

    return chunks


def print_chunks(query: str, chunks: list[dict]):
    print(f"\n=== Top {len(chunks)} results for: \"{query}\" ===\n")
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        print(f"--- Result {i + 1} (similarity: {chunk['similarity']:.3f}) ---")
        print(f"Drug: {meta['drug_name']} | Type: {meta['document_type']} | Ref: {meta['reference_id']}")
        print(f"Source: {meta['source_pdf']}")
        print(chunk["text"][:500])
        print()


def main():
    parser = argparse.ArgumentParser(description="Query ARIA memo chunks")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("-n", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--drug", help="Filter by drug_key (partial match)")
    parser.add_argument("--doctype", help="Filter by document_type (exact match)")
    args = parser.parse_args()

    chunks = retrieve(args.query, n=args.n, drug=args.drug, doctype=args.doctype)
    print_chunks(args.query, chunks)


if __name__ == "__main__":
    main()
