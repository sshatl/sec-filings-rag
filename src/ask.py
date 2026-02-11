import os
import re
from typing import List
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

from rag.mmr import mmr_select
from rag.formatting import format_citation, format_sources
from rag.filters import build_where, infer_section_contains

# ---------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------

load_dotenv()

CHROMA_DIR = "chroma"
COLLECTION = "sec_filings_openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

# System prompt enforces strict source-grounded answers
SYSTEM = """You are a careful assistant answering questions using ONLY the provided sources.
Rules:
- Use ONLY facts found in the sources.
- If the sources do not contain the answer, say: "Not found in provided sources."
- Cite sources using [1], [2], ... after the relevant sentences.
- Do not invent numbers, dates, or claims not present in sources.
"""


# ---------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------

def main(
    question: str,
    k: int = 5,
    ticker: str | None = None,
    form: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    section: str | None = None,
    section_contains: str | None = None,
    mmr: bool = True,
    mmr_lambda: float = 0.7,
):
    """End-to-end RAG: query → retrieve → (optional) filter → (optional) MMR → answer."""
    # Auto section routing if user did not specify filters
    auto_section = None
    if not section and not section_contains:
        auto_section = infer_section_contains(question)
        section_contains = auto_section

    base_where = build_where(ticker, form, min_date, max_date, section)

    # Setup clients
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma.get_collection(name=COLLECTION)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    # Embed the question
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

    # Over-fetch to allow post-filtering / MMR reranking
    prefetch_k = max(k * 8, 40) if (section_contains or mmr) else k

    # Query Chroma with metadata filter (if any)
    query_kwargs = dict(
        query_embeddings=[q_emb],
        n_results=prefetch_k,
        include=["documents", "metadatas", "distances"],
    )
    if base_where is not None:
        query_kwargs["where"] = base_where

    res = col.query(**query_kwargs)

    ids = res.get("ids", [[]])
    if not ids or not ids[0]:
        print("\n=== ANSWER ===\nNot found in provided sources.")
        print("\n=== SOURCES ===\n(no results)")
        return

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Post-filter by section substring (case-insensitive)
    if section_contains:
        needle = section_contains.strip().lower()
        filtered = [
            (d, m, s)
            for d, m, s in zip(docs, metas, dists)
            if needle in str(m.get("section", "")).lower()
        ]
        if not filtered:
            print("\n=== ANSWER ===\nNot found in provided sources.")
            print("\n=== SOURCES ===\n(no results after section filter)")
            return

        docs = [x[0] for x in filtered]
        metas = [x[1] for x in filtered]
        dists = [x[2] for x in filtered]

    # Final top-k selection (MMR or simple top-k)
    if mmr:
        docs, metas, dists = mmr_select(docs, metas, dists, k, mmr_lambda)
    else:
        docs = docs[:k]
        metas = metas[:k]
        dists = dists[:k]

    # Build LLM prompt
    sources_text = format_sources(docs, metas)
    user_prompt = f"""Question: {question}
Sources:
{sources_text}

Answer:"""

    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    # Print answer + citations
    print("\n=== ANSWER ===\n")
    if auto_section:
        print(f"(auto section routing: {auto_section})\n")
    print(resp.choices[0].message.content)

    print("\n=== SOURCES ===\n")
    for i, m in enumerate(metas, 1):
        print(format_citation(i, m))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ask questions over SEC filings via RAG (Chroma + OpenAI).")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ticker", type=str, default=None, help="e.g. TSLA")
    p.add_argument("--form", type=str, default=None, help="e.g. 10-K or 10-Q")
    p.add_argument("--min_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--max_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--section", type=str, default=None, help="exact section name (metadata)")
    p.add_argument("--section_contains", type=str, default=None, help="substring match, e.g. 'Risk Factors'")

    # MMR controls
    p.add_argument("--mmr", dest="mmr", action="store_true", help="Enable MMR reranking (default).")
    p.add_argument("--no_mmr", dest="mmr", action="store_false", help="Disable MMR reranking.")
    p.set_defaults(mmr=True)

    p.add_argument("--mmr_lambda", type=float, default=0.7, help="MMR lambda (0..1). Higher => more relevance.")

    args = p.parse_args()

    main(
        question=args.question,
        k=args.k,
        ticker=args.ticker,
        form=args.form,
        min_date=args.min_date,
        max_date=args.max_date,
        section=args.section,
        section_contains=args.section_contains,
        mmr=args.mmr,
        mmr_lambda=args.mmr_lambda,
    )
