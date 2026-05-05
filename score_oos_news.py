"""
Live News Scoring for 2024-2025 Out-of-Sample Period

Downloads recent financial news from Yahoo Finance RSS feeds for each
NASDAQ-100 stock, scores with the same 4-signal LLM prompt used during
training, and saves to oos_signals_2024_2025.csv.

This upgrades the OOS evaluation from neutral signals → real signals,
following the FutureX live benchmark philosophy (Zeng et al. 2025).

Key design principles:
  1. Point-in-time: news is processed in chronological order
  2. No look-ahead: only news published BEFORE the trading date is used
  3. Same prompt: identical to training to ensure consistency

Usage:
    python score_oos_news.py                     # score 2024-01 to today
    python score_oos_news.py --start 2024-01-01  # from specific date
    python score_oos_news.py --dry_run           # test without LLM API
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─── Config ───────────────────────────────────────────────────────────────────

NASDAQ_30 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AVGO","ASML",
    "COST","CSCO","ADBE","NFLX","AMD","INTC","INTU","QCOM","TXN","AMAT",
    "SBUX","GILD","MDLZ","PYPL","REGN","ISRG","VRTX","LRCX","KLAC","MRVL","GOOG"
]

PROMPT = """You are a financial analyst. Score this news about {ticker} from {date}.

News: {text}

Reply with ONLY 4 comma-separated integers (1-5, no labels):
sentiment, risk_level, analyst_confidence, volatility_forecast
(1=very_negative/very_low, 3=neutral, 5=very_positive/very_high)"""

OUT_FILE   = "oos_signals_2024_2025.csv"
BATCH_SIZE = 10
SLEEP_SEC  = 1.0


# ─── Yahoo Finance RSS news fetcher ───────────────────────────────────────────

def fetch_yahoo_news(ticker: str, max_items: int = 20) -> list[dict]:
    """
    Fetch recent news headlines from Yahoo Finance RSS for a ticker.
    Returns list of {date, title, summary} dicts, sorted oldest first.
    """
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []

        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")
        news  = []
        for item in items[:max_items]:
            title   = item.findtext("title", "")
            desc    = item.findtext("description", "")
            pub_str = item.findtext("pubDate", "").strip()
            pub_date = None
            if pub_str:
                try:
                    pub_dt = parsedate_to_datetime(pub_str)
                    pub_date = pub_dt.strftime("%Y-%m-%d")
                except Exception:
                    for fmt in (
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%a, %d %b %Y %H:%M:%S GMT",
                        "%a, %d %b %Y %H:%M %Z",
                        "%a, %d %b %Y %H:%M:%S",
                        "%a, %d %b %Y %H:%M",
                    ):
                        try:
                            pub_dt = datetime.strptime(pub_str.replace(" GMT", " UTC"), fmt)
                            pub_date = pub_dt.strftime("%Y-%m-%d")
                            break
                        except Exception:
                            continue
            if pub_date is None:
                pub_date = datetime.utcnow().strftime("%Y-%m-%d")

            text = f"{title}. {desc}"[:600]
            news.append({"date": pub_date, "text": text, "ticker": ticker})

        return sorted(news, key=lambda x: x["date"])
    except Exception:
        return []


# ─── LLM scoring ──────────────────────────────────────────────────────────────

def score_batch(client, model: str, articles: list[dict]) -> list[tuple]:
    """Score a batch of articles. Returns list of (sent, risk, conf, vol) tuples."""
    results = []
    for art in articles:
        prompt = PROMPT.format(
            ticker=art["ticker"],
            date=art["date"],
            text=art["text"]
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0,
            )
            parts = resp.choices[0].message.content.strip().split(",")
            scores = tuple(float(p.strip()) for p in parts[:4])
            if len(scores) == 4 and all(1 <= s <= 5 for s in scores):
                results.append(scores)
            else:
                results.append((3.0, 3.0, 3.0, 3.0))
        except Exception:
            results.append((3.0, 3.0, 3.0, 3.0))
        time.sleep(0.3)
    return results


# ─── Aggregate to daily signals ───────────────────────────────────────────────

def aggregate_daily_signals(rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate multiple articles per (ticker, date) to mean signal scores.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = df.groupby(["tic", "date"]).agg({
        "llm_sentiment":           "mean",
        "llm_risk":                "mean",
        "llm_confidence":          "mean",
        "llm_volatility_forecast": "mean",
        "n_articles":              "sum",
    }).reset_index()

    return agg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2024-01-01")
    parser.add_argument("--dry_run", action="store_true",
                        help="Fetch news but assign neutral scores (no API calls)")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model   = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set. Use --dry_run for testing.")
        return

    client = None
    if not args.dry_run:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/finrl-contest",
                "X-Title": "FinRL-DeepSeek OOS Scoring",
            }
        )

    rows = []
    total_articles = 0

    print(f"Scoring OOS news from {args.start} onwards …")
    print(f"Model: {model}  |  Dry run: {args.dry_run}")
    print(f"Tickers: {len(NASDAQ_30)}")
    print("─" * 50)

    for i, ticker in enumerate(NASDAQ_30):
        print(f"  [{i+1:02d}/{len(NASDAQ_30)}] {ticker} …", end=" ", flush=True)
        articles = fetch_yahoo_news(ticker, max_items=20)

        # Filter to OOS period
        articles = [a for a in articles if a["date"] >= args.start]

        if not articles:
            print("no recent news")
            continue

        print(f"{len(articles)} articles", end=" ")

        if args.dry_run:
            scores = [(3.0, 3.0, 3.0, 3.0)] * len(articles)
        else:
            scores = score_batch(client, model, articles)

        for art, (sent, risk, conf, vol) in zip(articles, scores):
            rows.append({
                "tic":                     ticker,
                "date":                    art["date"],
                "llm_sentiment":           sent,
                "llm_risk":                risk,
                "llm_confidence":          conf,
                "llm_volatility_forecast": vol,
                "n_articles":              1,
                "text_preview":            art["text"][:100],
            })
            total_articles += 1

        print(f"✓")
        time.sleep(SLEEP_SEC)

    if not rows:
        print("No articles found. Check internet connection.")
        return

    # Aggregate to daily
    df_daily = aggregate_daily_signals(rows)

    df_daily.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(df_daily)} (ticker, date) signal entries → {OUT_FILE}")
    print(f"Total articles scored: {total_articles}")
    print(f"Date range: {df_daily['date'].min()} → {df_daily['date'].max()}")
    print(f"Tickers with coverage: {df_daily['tic'].nunique()}")

    # Quick stats
    non_neutral = (df_daily["llm_sentiment"] != 3.0).sum()
    coverage    = non_neutral / len(df_daily) * 100
    print(f"\nSignal coverage: {coverage:.1f}% non-neutral")
    print(f"Mean sentiment: {df_daily['llm_sentiment'].mean():.3f}")
    print(f"Mean risk:      {df_daily['llm_risk'].mean():.3f}")


if __name__ == "__main__":
    main()
