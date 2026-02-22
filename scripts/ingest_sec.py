import requests
import os
import time
import argparse
import re

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

    
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
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