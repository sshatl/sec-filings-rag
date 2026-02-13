import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings_openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")


def main(q: str, k: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(name=COLLECTION)

    oai = OpenAI(api_key=OPENAI_API_KEY)
    emb = oai.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = emb.data[0].embedding

    res = col.query(query_embeddings=[q_vec], n_results=k)

    for i in range(k):
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        print("\n---", i + 1, meta)
        print(doc[:900], "...")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("q")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    main(args.q, args.k)
