import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
TEXT_DIR = Path("data/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def date_to_int(d: str | None) -> int | None:
    if not d:
        return None
    try:
        return int(str(d).replace("-", ""))
    except ValueError:
        return None

def clean_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    out = []

    for ln in lines:
        if not ln:
            continue

        low = ln.lower()

        # typical headers/footers to remove
        if low in {"table of contents"}:
            continue

        # page numbers like "29"
        if re.fullmatch(r"\d{1,3}", ln):
            continue

        # table noise: lines with very few letters
        letters = sum(ch.isalpha() for ch in ln)
        if letters < 5 and len(ln) > 10:
            continue

        out.append(ln)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

SECTION_PATTERNS = [
    (r"\bITEM\s+1A\b", "Item 1A - Risk Factors"),
    (r"\bITEM\s+1\b",  "Item 1 - Business / Financial Statements"),
    (r"\bITEM\s+2\b",  "Item 2 - MD&A / Properties"),
    (r"\bITEM\s+3\b",  "Item 3 - Legal Proceedings"),
    (r"\bITEM\s+7A\b", "Item 7A - Market Risk"),
    (r"\bITEM\s+7\b",  "Item 7 - MD&A"),
    (r"\bITEM\s+8\b",  "Item 8 - Financial Statements"),
    (r"\bITEM\s+9A\b", "Item 9A - Controls and Procedures"),
    (r"\bITEM\s+9\b",  "Item 9 - Changes and Disagreements"),
]

def normalize_text_for_sections(text: str) -> str:
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return t

def split_into_sections(text: str):
    """
    Returns list of dicts: [{"section": str, "text": str}]
    If no sections found -> one section "Unknown".
    """
    t = normalize_text_for_sections(text)

    # find anchors
    anchors = []
    for pattern, name in SECTION_PATTERNS:
        for m in re.finditer(pattern, t, flags=re.IGNORECASE):
            anchors.append((m.start(), name, m.group(0)))

    if not anchors:
        return [{"section": "Unknown", "text": text.strip()}]

    # sort and de-dup anchors by position, prefer longer item labels if same position
    anchors.sort(key=lambda x: x[0])
    dedup = []
    last_pos = None
    for pos, name, raw in anchors:
        if last_pos == pos:
            # skip duplicates at same position
            continue
        dedup.append((pos, name))
        last_pos = pos
    anchors = dedup

    # slice
    out = []
    for idx, (start, name) in enumerate(anchors):
        end = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(t)
        chunk_txt = t[start:end].strip()
        if chunk_txt:
            out.append({"section": name, "text": chunk_txt})

    # if first anchor starts too late, keep the preface
    first_start = anchors[0][0]
    if first_start > 200:
        pre = t[:first_start].strip()
        if pre:
            out.insert(0, {"section": "Preface", "text": pre})

    return out

def html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")

    # remove scripts/styles/noscript 
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")

    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove noisy lines
    text = clean_lines(text)

    return text.strip()


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200):
    """
    split text into chunks of max_chars with overlap
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)

        start = max(0, end - overlap)
        if end == n:
            break

    return chunks


def main(max_chars: int = 2500, overlap: int = 200):
    meta_path = Path("data/filings_meta.json")
    filings = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = []

    for item in filings:
        fpath = Path(item["file"])
        html = fpath.read_bytes()

        text = html_to_text(html)

        sections = split_into_sections(text)
        base_id = fpath.stem

        global_chunk_idx = 0
        for s_idx, sec in enumerate(sections):
            sec_name = sec["section"]
            sec_text = sec["text"]

            sec_chunks = chunk_text(sec_text, max_chars=max_chars, overlap=overlap)

            for local_i, ch in enumerate(sec_chunks):
                rows.append(
                    {
                        "id": f"{base_id}__chunk{global_chunk_idx:04d}",
                        "text": ch,
                        "meta": {
                            "ticker": item.get("ticker") or base_id.split("__")[0],
                            "form": item.get("form"),
                            "reportDate": item.get("reportDate"),
                            "reportDate_int": date_to_int(item.get("reportDate")),
                            "accession": item.get("accession"),
                            "acc_nodash": item.get("acc_nodash"),
                            "cik_nozero": item.get("cik_nozero"),
                            "primaryDocument": item.get("primaryDocument"),
                            "edgar_url": item.get("edgar_url"),
                            "source_file": str(fpath),
                            "chunk_index": global_chunk_idx,

                            # NEW:
                            "section": sec_name,
                            "section_index": s_idx,
                            "section_chunk_index": local_i,
                        },
                    }
                )
                global_chunk_idx += 1

    Path("data/chunks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )

    print(f"Chunks written: {len(rows)} → data/chunks.jsonl")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max_chars", type=int, default=2500)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    main(args.max_chars, args.overlap)
