"""Step 4: Retrieve chunks from ChromaDB by semantic similarity."""
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = Path("chroma_db")


def main():
    parser = argparse.ArgumentParser(description="Query ARIA memo chunks")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("-n", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--drug", help="Filter by drug_key (exact match)")
    parser.add_argument("--doctype", help="Filter by document_type (exact match)")
    args = parser.parse_args()

    # Load same model used in Step 3
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to existing DB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("aria_memos")

    # Embed the query
    query_embedding = model.encode(args.query).tolist()

# Build metadata filter
    where = None
    conditions = []
    if args.drug:
        # Find all drug_keys containing the search term
        import json
        matching_keys = set()
        for line in open("chunks.jsonl", encoding="utf-8"):
            c = json.loads(line)
            dk = c["metadata"]["drug_key"].lower()
            if args.drug.lower() in dk:
                matching_keys.add(c["metadata"]["drug_key"])
        if matching_keys:
            print(f"Matched drug(s): {', '.join(sorted(matching_keys))}")
            conditions.append({"drug_key": {"$in": list(matching_keys)}})
        else:
            print(f"No drugs matched '{args.drug}'")
            return
    if args.doctype:
        conditions.append({"document_type": args.doctype})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Print results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\n=== Top {len(docs)} results for: \"{args.query}\" ===\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        sim = 1 - dist  # cosine distance → similarity
        print(f"--- Result {i + 1} (similarity: {sim:.3f}) ---")
        print(f"Drug: {meta['drug_name']} | Type: {meta['document_type']} | Ref: {meta['reference_id']}")
        print(f"Source: {meta['source_pdf']}")
        print(doc[:500])
        print()


if __name__ == "__main__":
    main()