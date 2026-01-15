import chromadb

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings"


def main(q: str, k: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(name=COLLECTION)

    res = col.query(query_texts=[q], n_results=k)

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
