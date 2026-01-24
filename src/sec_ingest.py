import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
TEXT_DIR = Path("data/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


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

        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        base_id = fpath.stem

        for i, ch in enumerate(chunks):
            rows.append(
                {
                    "id": f"{base_id}__chunk{i:04d}",
                    "text": ch,
                    "meta": {
                        "ticker": item.get("ticker") or base_id.split("__")[0],
                        "form": item.get("form"),
                        "reportDate": item.get("reportDate"),
                        "accession": item.get("accession"),
                        "acc_nodash": item.get("acc_nodash"),
                        "cik_nozero": item.get("cik_nozero"),
                        "primaryDocument": item.get("primaryDocument"),
                        "edgar_url": item.get("edgar_url"),
                        "source_file": str(fpath),
                        "chunk_index": i,
                    },

                }
            )

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
