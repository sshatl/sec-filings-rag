import os
import json
import time
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEC_UA = os.getenv("SEC_USER_AGENT", "").strip()
if not SEC_UA:
    raise RuntimeError("Set SEC_USER_AGENT in .env (e.g., 'YourName your@email.com')")

HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean_cik(cik: str) -> str:
    cik = str(cik).strip()
    cik = re.sub(r"\D", "", cik)
    return cik.zfill(10)


def ticker_to_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    t = ticker.strip().upper()
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == t:
            return _clean_cik(row["cik_str"])
    raise ValueError(f"Ticker not found: {ticker}")


def list_filings(cik10: str, forms=("10-K", "10-Q"), limit=20):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()

    recent = j.get("filings", {}).get("recent", {})
    form = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    primary = recent.get("primaryDocument", [])
    report_date = recent.get("reportDate", [])

    out = []
    for f, acc, doc, rd in zip(form, accession, primary, report_date):
        if f in forms:
            out.append({"form": f, "accession": acc, "primaryDocument": doc, "reportDate": rd})
        if len(out) >= limit:
            break
    return out


def download_filing_html(cik10: str, acc: str, primary_doc: str, save_path: Path):
    cik_nozero = str(int(cik10))
    acc_nodash = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_nodash}/{primary_doc}"

    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    save_path.write_bytes(r.content)


def main(ticker: str, limit: int = 20):
    cik10 = ticker_to_cik(ticker)
    filings = list_filings(cik10, limit=limit)

    out_meta = []
    for f in filings:
        form = f["form"]
        acc = f["accession"]
        doc = f["primaryDocument"]
        rd = f["reportDate"] or "unknown_date"

        fname = f"{ticker.upper()}__{form}__{rd}__{acc.replace('-', '')}__{doc}"
        save_path = DATA_DIR / fname

        if not save_path.exists():
            download_filing_html(cik10, acc, doc, save_path)
            time.sleep(0.25)

        cik_nozero = str(int(cik10))
        acc_nodash = acc.replace("-", "")
        edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_nodash}/{doc}"

        out_meta.append({
            "ticker": ticker.upper(),
            "cik10": cik10,
            "cik_nozero": cik_nozero,
            "form": form,
            "accession": acc,
            "acc_nodash": acc_nodash,
            "primaryDocument": doc,
            "reportDate": rd,
            "edgar_url": edgar_url,
            "file": str(save_path),
        })


    Path("data").mkdir(exist_ok=True)
    Path("data/filings_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"Saved {len(out_meta)} filings. Meta: data/filings_meta.json")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    main(args.ticker, args.limit)
