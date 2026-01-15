
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

load_dotenv()

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings_openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")


def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # reset collection for repeatability
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.get_or_create_collection(name=COLLECTION)

    # load chunks
    ids, docs, metas = [], [], []
    with Path("data/chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            docs.append(row["text"])
            metas.append(row["meta"])

    oai = OpenAI(api_key=OPENAI_API_KEY)

    B = 64
    for start in range(0, len(ids), B):
        batch_ids = ids[start:start+B]
        batch_docs = docs[start:start+B]
        batch_metas = metas[start:start+B]

        # embeddings call
        emb = oai.embeddings.create(
            model=EMBED_MODEL,
            input=batch_docs
        )
        vectors = [e.embedding for e in emb.data]

        col.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=vectors,
        )
        print(f"Indexed {min(start+B, len(ids))}/{len(ids)}")

    print("Done. Total:", col.count())


if __name__ == "__main__":
    main()
