import os
import re
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


def format_citation(i: int, meta: dict) -> str:
    ticker = meta.get("ticker", "UNK")
    form = meta.get("form", "UNK")
    rd = meta.get("reportDate", "UNK")
    url = meta.get("edgar_url")
    chunk = meta.get("chunk_index")
    acc = meta.get("accession")

    if url:
        return f"[{i}] {ticker} {form} {rd} (chunk={chunk}, acc={acc})\n    EDGAR: {url}"
    src = meta.get("source_file", "unknown")
    return f"[{i}] {ticker} {form} {rd} (chunk={chunk}, acc={acc})\n    Source: {src}"


def format_sources(docs, metas, max_chars_per_source=1200):
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        snippet = (doc or "")[:max_chars_per_source].strip()
        header = (
            f"[{i}] ticker={meta.get('ticker')} form={meta.get('form')} "
            f"reportDate={meta.get('reportDate')} chunk={meta.get('chunk_index')} "
            f"section={meta.get('section')} edgar_url={meta.get('edgar_url')}"
        )
        blocks.append(header + "\n" + snippet)
    return "\n\n".join(blocks)


def date_to_int(d: str | None) -> int | None:
    if not d:
        return None
    try:
        return int(d.replace("-", ""))
    except ValueError:
        return None


def build_where(
    ticker: str | None,
    form: str | None,
    min_date: str | None,
    max_date: str | None,
    section: str | None,
) -> dict | None:
    clauses = []
    if ticker:
        clauses.append({"ticker": ticker.strip().upper()})
    if form:
        clauses.append({"form": form.strip().upper()})
    if min_date:
        mi = date_to_int(min_date)
        if mi is not None:
            clauses.append({"reportDate_int": {"$gte": mi}})
    if max_date:
        ma = date_to_int(max_date)
        if ma is not None:
            clauses.append({"reportDate_int": {"$lte": ma}})
    if section:
        clauses.append({"section": section.strip()})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_section_contains(question: str) -> str | None:
    """
    Heuristic router: maps question -> section substring.
    You can extend this safely.
    """
    q = _norm(question)

    # Strong signals
    rules = [
        (r"\brisk factors?\b|\bitem 1a\b|\brisks?\b", "Risk Factors"),
        (r"\bmd&a\b|\bmanagement(?:'s)? discussion\b|\bliquidity\b|\bcash flows?\b|\bresults of operations\b|\boutlook\b", "MD&A"),
        (r"\bmarket risk\b|\binterest rate\b|\bfx\b|\bforeign exchange\b|\bcommodity\b|\bderivative\b|\bsensitivity\b", "Market Risk"),
        (r"\bcontrols?\b|\bdisclosure controls\b|\binternal control\b|\bsox\b|\bitem 9a\b", "Controls and Procedures"),
        (r"\blegal proceedings\b|\blawsuit\b|\blitigation\b|\bitem 3\b", "Legal Proceedings"),
        (r"\bfinancial statements\b|\bbalance sheet\b|\bincome statement\b|\bcash flow statement\b|\bnotes to\b|\bitem 8\b", "Financial Statements"),
        (r"\bbusiness\b|\boperations\b|\bitem 1\b", "Business"),
        (r"\bproperties\b|\bfacilities\b|\bitem 2\b", "Properties"),
    ]

    for pat, sec in rules:
        if re.search(pat, q):
            return sec

    return None


def main(
    question: str,
    k: int,
    ticker: str | None,
    form: str | None,
    min_date: str | None,
    max_date: str | None,
    section: str | None,
    section_contains: str | None,
):
    # If user didn't provide section filters, auto-route
    auto_section_contains = None
    if not section and not section_contains:
        auto_section_contains = infer_section_contains(question)
        section_contains = auto_section_contains

    base_where = build_where(ticker, form, min_date, max_date, section)

    # retrieval setup
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma_client.get_collection(name=COLLECTION)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

    prefetch_k = max(k * 6, 30) if section_contains else k

    query_kwargs = {
        "query_embeddings": [q_emb],
        "n_results": prefetch_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if base_where is not None:
        query_kwargs["where"] = base_where

    res = col.query(**query_kwargs)

    if not res.get("ids") or not res["ids"] or not res["ids"][0]:
        print("\n=== ANSWER ===\n")
        print("Not found in provided sources.")
        print("\n=== SOURCES (top-k) ===\n")
        print("(no results)")
        return

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(docs)])[0]

    # If section_contains is set, keep only matches (but we prefetch more)
    if section_contains:
        needle = section_contains.strip().lower()
        filtered = []
        for doc, meta, dist in zip(docs, metas, dists):
            sec = str(meta.get("section", "")).lower()
            if needle in sec:
                filtered.append((doc, meta, dist))

        if not filtered:
            print("\n=== ANSWER ===\n")
            print("Not found in provided sources.")
            print("\n=== SOURCES (top-k) ===\n")
            if auto_section_contains:
                print(f"(no results after auto section routing: '{auto_section_contains}')")
            else:
                print("(no results after section filter)")
            return

        filtered = filtered[:k]
        docs = [x[0] for x in filtered]
        metas = [x[1] for x in filtered]
    else:
        docs = docs[:k]
        metas = metas[:k]

    sources_text = format_sources(docs, metas)

    # Optional: tell the model which section was targeted (helps grounding)
    section_hint = ""
    if section_contains:
        section_hint = f"\nTargeted section: {section_contains}\n"

    user = f"""Question: {question}
{section_hint}
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
    if auto_section_contains:
        print(f"(auto section routing: {auto_section_contains})\n")
    print(answer)

    print("\n=== SOURCES (top-k) ===\n")
    for i, meta in enumerate(metas, 1):
        print(format_citation(i, meta))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ticker", type=str, default=None, help="e.g. TSLA")
    p.add_argument("--form", type=str, default=None, help="e.g. 10-K or 10-Q")
    p.add_argument("--min_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--max_date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--section", type=str, default=None, help="exact section name (metadata)")
    p.add_argument("--section_contains", type=str, default=None, help="substring match, e.g. 'Risk Factors'")

    args = p.parse_args()

    main(
        args.question,
        args.k,
        args.ticker,
        args.form,
        args.min_date,
        args.max_date,
        args.section,
        args.section_contains,
    )
