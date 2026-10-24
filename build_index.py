import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "weather_knowledge.txt"
INDEX_FILE = BASE_DIR / "vector_index.json"

text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

# Simple chunking: one paragraph per chunk.
chunks = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

index = []

for chunk, item in zip(chunks, response.data):
    index.append({
        "text": chunk,
        "embedding": item.embedding
    })

INDEX_FILE.write_text(
    json.dumps(index),
    encoding="utf-8"
)

print(f"Indexed {len(index)} text chunks.")
print(f"Saved vector index to: {INDEX_FILE}")