import os
import re
from typing import List
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

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
# Output formatting helpers
# ---------------------------------------------------------------------

def format_citation(i: int, meta: dict) -> str:
    """Pretty-print a single source citation."""
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


def format_sources(docs, metas, max_chars_per_source=1200) -> str:
    """Prepare retrieved chunks for LLM context (with metadata headers)."""
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

# ---------------------------------------------------------------------
# Metadata utilities
# ---------------------------------------------------------------------

def date_to_int(d: str | None) -> int | None:
    """Convert YYYY-MM-DD to integer for numeric filtering."""
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
    """Build Chroma metadata filter (AND-combined)."""
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

# ---------------------------------------------------------------------
# Section routing heuristics
# ---------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize text for heuristic matching."""
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_section_contains(question: str) -> str | None:
    """Heuristic router: infers relevant SEC section based on question intent."""
    q = _norm(question)

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

# ---------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) helpers
# ---------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)


def _token_set(text: str, max_tokens: int = 600) -> set:
    """Lightweight tokenization for diversity scoring."""
    if not text:
        return set()
    toks = _TOKEN_RE.findall(text.lower())
    return set(toks[:max_tokens])


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    union = (a | b)
    return (len(a & b) / len(union)) if union else 0.0


def _dist_to_relevance(dist: float | None) -> float:
    """Convert vector distance to relevance score (smaller dist => higher relevance)."""
    if dist is None:
        return 0.0
    return 1.0 / (1.0 + float(dist))


def mmr_select(
    docs: List[str],
    metas: List[dict],
    dists: List[float | None],
    k: int,
    lamb: float = 0.7,
):
    """MMR selection to balance relevance and diversity among retrieved chunks."""
    n = len(docs)
    if n <= k:
        return docs, metas, dists

    relevance = [_dist_to_relevance(d) for d in dists]
    toksets = [_token_set(doc) for doc in docs]

    selected: List[int] = []
    candidates = list(range(n))

    # Start with the most relevant chunk
    first = max(candidates, key=lambda i: relevance[i])
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        def score(i: int) -> float:
            max_sim = max(_jaccard(toksets[i], toksets[j]) for j in selected)
            return lamb * relevance[i] - (1 - lamb) * max_sim

        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)

    return (
        [docs[i] for i in selected],
        [metas[i] for i in selected],
        [dists[i] for i in selected],
    )

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
