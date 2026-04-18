# peek_chunks.py
import json

# chunks = [json.loads(line) for line in open("chunks.jsonl")]
chunks = [json.loads(line) for line in open("chunks.jsonl", encoding="utf-8")]
by_size = sorted(chunks, key=lambda x: x["word_count"])

print("SMALLEST 3:")
for c in by_size[:3]:
    print(f"  {c['word_count']} words | {c['chunk_id']}")
    print(f"  {c['text'][:100]}")
    print()

print("LARGEST 3:")
for c in by_size[-3:]:
    print(f"  {c['word_count']} words | {c['chunk_id']}")
    print(f"  {c['text'][:100]}")
    print()