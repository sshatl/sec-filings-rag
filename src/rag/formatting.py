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