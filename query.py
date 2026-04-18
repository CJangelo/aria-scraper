"""
query.py — Retrieve ARIA memo chunks and synthesize an answer via LLM.

Uses LiteLLM so the model is swappable: Claude, GPT-4, Ollama, etc.
Set the model via --model or the LLM_MODEL environment variable.

Usage:
    uv run python query.py "what are the safety concerns for tirzepatide"
    uv run python query.py "..." --model claude-sonnet-4-6
    uv run python query.py "..." --model ollama/llama3
    uv run python query.py "..." --no-generate        # retrieval only
    uv run python query.py "..." --drug zepbound --doctype aria_sufficiency
"""

import argparse
import os

import litellm

from retrieve import retrieve, print_chunks

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an analyst reviewing FDA ARIA (Accelerated Review of Information \
and Analysis) drug review memos. Answer the user's question using only the provided excerpts. \
Be concise and precise. Cite sources inline as [1], [2], etc., matching the excerpt numbers. \
If the excerpts do not contain enough information to answer, say so."""


def build_user_message(query: str, chunks: list[dict]) -> str:
    lines = [f"Question: {query}", "", "Excerpts:"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[{i}] Drug: {meta.get('drug_name', '?')} | "
            f"Type: {meta.get('document_type', '?')} | "
            f"Pages: {meta.get('page_range', '?')} | "
            f"Source: {meta.get('source_pdf', '?')}"
        )
        lines.append(header)
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def generate(query: str, chunks: list[dict], model: str) -> str:
    user_message = build_user_message(query, chunks)
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Query ARIA memos with LLM synthesis.")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("-n", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    parser.add_argument("--drug", help="Filter by drug key (partial match)")
    parser.add_argument("--doctype", help="Filter by document_type (exact match)")
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        help="LiteLLM model string (default: LLM_MODEL env var or claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM call and print raw chunks only",
    )
    args = parser.parse_args()

    chunks = retrieve(args.query, n=args.n, drug=args.drug, doctype=args.doctype)

    if not chunks:
        print("No results found.")
        return

    if args.no_generate:
        print_chunks(args.query, chunks)
        return

    print(f"\nRetrieved {len(chunks)} chunks. Generating answer with {args.model}...\n")

    answer = generate(args.query, chunks, args.model)

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("\n--- Sources ---")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        print(
            f"[{i}] {meta.get('drug_name', '?')} | "
            f"{meta.get('document_type', '?')} | "
            f"Pages {meta.get('page_range', '?')} | "
            f"{meta.get('source_pdf', '?')}"
        )


if __name__ == "__main__":
    main()
