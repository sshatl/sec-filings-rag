# ---------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) helpers
# ---------------------------------------------------------------------


import re
from typing import List

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
