# ---------------------------------------------------------------------
# Metadata utilities
# ---------------------------------------------------------------------

import re


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
