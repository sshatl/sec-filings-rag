import json
from pathlib import Path

import chromadb

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

  
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.get_or_create_collection(name=COLLECTION)

    ids, docs, metas = [], [], []
    with Path("data/chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            docs.append(row["text"])
            metas.append(row["meta"])

    B = 256
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i:i+B],
            documents=docs[i:i+B],
            metadatas=metas[i:i+B],
        )

    print("Indexed:", col.count())


if __name__ == "__main__":
    main()
