import requests
import os
import time
import argparse
import re
import html
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "financial-rag nishantkumbhar812@gmail.com"}
BASE_URL = "https://data.sec.gov/submissions/CIK{}.json"
SAVE_DIR = "data/raw"

def get_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")

def get_filings(cik: str, filing_type: str) -> list:
    url = BASE_URL.format(cik)
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()
    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form == filing_type:
            results.append({
                "accession_dashed": filings["accessionNumber"][i],               # 0000320193-23-000106
                "accession_nodash": filings["accessionNumber"][i].replace("-", ""),  # 000032019323000106
                "date": filings["filingDate"][i],
                "form": form,
            })
    return results

def download_filing(cik: str, accession_dashed: str, accession_nodash: str) -> str:
    txt_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{accession_dashed}.txt"
    print(f"  Fetching: {txt_url}")
    resp = requests.get(txt_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

    
def clean_text(raw_text: str) -> str:
    """
    Cleans SEC EDGAR filing text for semantic retrieval.
    Removes HTML, XBRL, boilerplate, headers, and useless sections.
    """

    # 1️⃣ Decode HTML entities (&#8217; etc.)
    text = html.unescape(raw_text)

    # 2️⃣ Keep only main 10-K document (ignore exhibits)
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, re.DOTALL | re.IGNORECASE)
    if documents:
        # Usually first DOCUMENT block is main filing
        text = documents[0]

    # 3️⃣ Remove XBRL tags completely
    text = re.sub(r"<[^>]+>", " ", text)

    # 4️⃣ Remove SEC header junk before ITEM 1
    item_start = re.search(r"ITEM\s+1\.", text, re.IGNORECASE)
    if item_start:
        text = text[item_start.start():]

    # 5️⃣ Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # 6️⃣ Remove checkbox artifacts
    text = re.sub(r"[☒☐]", " ", text)

    # 7️⃣ Remove repetitive boilerplate phrases
    boilerplate_patterns = [
        r"Securities registered pursuant to Section 12\(b\).*?",
        r"Indicate by check mark whether.*?",
        r"The Nasdaq Stock Market LLC",
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text.strip()

def scrape(ticker: str, filing_type: str = "10-K", max_filings: int = 3):
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Looking up CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"CIK: {cik}")

    print(f"Fetching {filing_type} filings...")
    filings = get_filings(cik, filing_type)[:max_filings]
    print(f"Found {len(filings)} filings")

    for filing in filings:
        filename = f"{SAVE_DIR}/{ticker}_{filing['form']}_{filing['date']}.txt"
        if os.path.exists(filename):
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading {filing['form']} from {filing['date']}...")
        try:
            text = download_filing(cik, filing["accession_dashed"], filing["accession_nodash"])
            text = clean_text(text)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {filename} ({len(text):,} chars)")
        except Exception as e:
            print(f"Failed: {e}")

        time.sleep(0.5)

    print("\nDone! Files saved to data/raw/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--filing", default="10-K", help="10-K or 10-Q")
    parser.add_argument("--max", type=int, default=3, help="Number of filings to download")
    args = parser.parse_args()

    scrape(args.ticker, args.filing, args.max)