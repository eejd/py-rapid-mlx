#!/usr/bin/env python3
"""Read an Alphabet quote from its public Google Finance page."""

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone


def parse_quote(html, ticker):
    match = re.search(
        r'<div class="gO24Ff">(Alphabet Inc Class [AC])</div>.{0,500}?'
        r'<span jsname="Pdsbrc"[^>]*><span>(\$[\d,]+\.\d{2})</span>',
        html,
    )
    if not match:
        raise ValueError(f"No verified {ticker} quote found on the source page")
    company, price = match.groups()
    expected_class = "A" if ticker == "GOOGL" else "C"
    if not company.endswith(expected_class):
        raise ValueError(f"Source page returned the wrong share class for {ticker}")
    return company, price


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) == 2 else ""
    if ticker not in {"GOOGL", "GOOG"}:
        raise SystemExit("Usage: goose-finance-quote.py GOOGL|GOOG")
    url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=8) as response:
        html = response.read().decode("utf-8")
    company, price = parse_quote(html, ticker)
    print(json.dumps({
        "ticker": ticker,
        "company": company,
        "price": price,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": url,
    }))


if __name__ == "__main__":
    main()
