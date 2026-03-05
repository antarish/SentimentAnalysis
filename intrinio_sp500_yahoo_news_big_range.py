# intrinio_sp500_yahoo_news_big_range.py
# pip install requests pandas python-dateutil

import os
import time
import random
from datetime import date
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd

API_KEY = os.getenv("INTRINIO_API_KEY")  # or set API_KEY = "YOUR_KEY"
BASE_URL = "https://api-v2.intrinio.com"

# Set your biggest feasible range here
START_DATE = "2009-10-01"
END_DATE = date.today().isoformat()  # today

SP500_CSV = r"sp500_constituents.csv"  # CSV with column: Symbol
OUT_CSV = f"sp500_yahoo_finance_news_{START_DATE}_to_{END_DATE}.csv"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "intrinio-news-puller/1.0"})


def intrinio_get(path: str, params: dict) -> dict:
    params = dict(params or {})
    params["api_key"] = API_KEY
    r = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=30)

    # basic backoff for rate limiting / transient issues
    if r.status_code in (429, 500, 502, 503, 504):
        raise requests.HTTPError(f"{r.status_code} transient", response=r)

    r.raise_for_status()
    return r.json()


def load_sp500_tickers(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns:
        raise ValueError(f"CSV must have a 'Symbol' column. Found columns: {list(df.columns)}")

    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(tickers)


def month_windows(start_iso: str, end_iso: str):
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)

    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        win_start = max(start, cur)
        win_end = min(end, nxt - relativedelta(days=1))
        yield win_start.isoformat(), win_end.isoformat()
        cur = nxt


def fetch_company_news_yahoo(ticker: str, start_date: str, end_date: str) -> list[dict]:
    rows = []
    next_page = None

    while True:
        params = {
            "specific_source": "yahoo",
            "start_date": start_date,
            "end_date": end_date,
            "page_size": 100,
        }
        if next_page:
            params["next_page"] = next_page

        data = intrinio_get(f"/companies/{ticker}/news", params)
        news_items = data.get("news") or []

        for a in news_items:
            rows.append(
                {
                    "ticker": ticker,
                    "window_start": start_date,
                    "window_end": end_date,
                    "publication_date": a.get("publication_date"),
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "summary": a.get("summary"),
                    "source": a.get("source"),
                    "intrinio_news_id": a.get("id"),
                }
            )

        next_page = data.get("next_page")
        if not next_page:
            break

        time.sleep(0.12 + random.random() * 0.12)

    return rows


def append_rows_to_csv(rows: list[dict], path: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def call_with_backoff(fn, max_tries: int = 6):
    for attempt in range(max_tries):
        try:
            return fn()
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None

            if status not in (429, 500, 502, 503, 504) and "transient" not in str(e):
                raise

            sleep_s = min(60, (2 ** attempt) + random.random())
            time.sleep(sleep_s)

    raise RuntimeError("Failed after retries (rate limit or persistent error).")


def dedupe_final_csv(path: str):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.drop_duplicates(subset=["ticker", "intrinio_news_id", "url", "title"])
    df = df.sort_values(["publication_date", "ticker", "title"])
    df.to_csv(path, index=False)


def main():
    if not API_KEY:
        raise RuntimeError("Missing INTRINIO_API_KEY. Set it as an environment variable or hardcode API_KEY.")

    tickers = load_sp500_tickers(SP500_CSV)
    windows = list(month_windows(START_DATE, END_DATE))

    print(f"Tickers: {len(tickers)}")
    print(f"Monthly windows: {len(windows)}")
    print(f"Output: {OUT_CSV}")

    for w_i, (w_start, w_end) in enumerate(windows, 1):
        print(f"\n=== Window {w_i}/{len(windows)}: {w_start} to {w_end} ===")

        for i, tkr in enumerate(tickers, 1):
            def fetch():
                return fetch_company_news_yahoo(tkr, w_start, w_end)

            try:
                rows = call_with_backoff(fetch)
                append_rows_to_csv(rows, OUT_CSV)
                print(f"[{i}/{len(tickers)}] {tkr}: +{len(rows)}")
            except Exception as e:
                print(f"[{i}/{len(tickers)}] {tkr}: error ({e}); skipping.")

            # pacing across tickers
            if i % 25 == 0:
                time.sleep(0.8)
            else:
                time.sleep(0.08 + random.random() * 0.15)

    dedupe_final_csv(OUT_CSV)
    print("\nDone (deduped).")


if __name__ == "__main__":
    main()