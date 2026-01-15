import os
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

load_dotenv()

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings_openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

SYSTEM = """You are a careful assistant answering questions using ONLY the provided sources.
Rules:
- Use ONLY facts found in the sources.
- If the sources do not contain the answer, say: "Not found in provided sources."
- Cite sources using [1], [2], ... after the relevant sentences.
- Do not invent numbers, dates, or claims not present in sources.
"""

def format_sources(docs, metas, max_chars_per_source=1200):
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        snippet = doc[:max_chars_per_source].strip()
        header = (
            f"[{i}] ticker={meta.get('ticker')} form={meta.get('form')} "
            f"reportDate={meta.get('reportDate')} chunk={meta.get('chunk_index')} "
            f"source_file={meta.get('source_file')}"
        )
        blocks.append(header + "\n" + snippet)
    return "\n\n".join(blocks)

def main(question: str, k: int = 5):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma_client.get_collection(name=COLLECTION)

    oai = OpenAI(api_key=OPENAI_API_KEY)

    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    sources_text = format_sources(docs, metas)

    user = f"""Question: {question}

Sources:
{sources_text}

Answer:"""

    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES (top-k) ===\n")
    for i, meta in enumerate(metas, 1):
        print(f"[{i}] {meta.get('source_file')} (form={meta.get('form')}, date={meta.get('reportDate')}, chunk={meta.get('chunk_index')})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    main(args.question, args.k)
