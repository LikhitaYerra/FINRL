"""
Merge real LLM signals from multi_signal_nasdaq_news.csv into the
train/trade price data CSVs.

Strategy:
  For each (ticker, date) in price data:
    - Look up all scored news rows for that ticker on that date
    - Take the mean of llm_sentiment, llm_risk, llm_confidence, llm_volatility_forecast
    - If no articles found for that date, look back up to LOOKBACK_DAYS for latest signal
    - If still nothing, keep neutral (3.0)
"""

import pandas as pd
import numpy as np

LOOKBACK_DAYS  = 3       # days to look back for most-recent signal if no same-day match
NEUTRAL        = 3.0
SIGNAL_COLS    = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]


def build_signal_lookup(news_df: pd.DataFrame) -> dict:
    """
    Build a dict: {(ticker, date_str) -> {signal: mean_value}}
    Only rows with non-null signals are included.
    """
    scored = news_df.dropna(subset=["llm_sentiment"]).copy()
    scored["Date"] = pd.to_datetime(scored["Date"]).dt.strftime("%Y-%m-%d")
    scored = scored.rename(columns={"Stock_symbol": "tic"})

    grouped = (
        scored
        .groupby(["tic", "Date"])[SIGNAL_COLS]
        .mean()
        .round(4)
        .reset_index()
    )

    lookup = {}
    for _, row in grouped.iterrows():
        lookup[(row["tic"], row["Date"])] = {c: row[c] for c in SIGNAL_COLS}

    print(f"  Signal lookup: {len(lookup):,} unique (ticker, date) pairs")
    return lookup


def fill_signals(price_df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """
    Fill llm_* columns in price_df using lookup, with LOOKBACK_DAYS fallback.
    """
    df = price_df.copy()
    dates = sorted(df["date"].unique())
    date_series = pd.to_datetime(dates)

    filled_exact = 0
    filled_lookback = 0
    kept_neutral = 0

    for col in SIGNAL_COLS:
        df[col] = NEUTRAL  # reset to neutral first

    for _, row in df.iterrows():
        tic  = row["tic"]
        date = row["date"]
        idx  = _

        # 1. Exact match
        if (tic, date) in lookup:
            for col in SIGNAL_COLS:
                df.at[idx, col] = lookup[(tic, date)][col]
            filled_exact += 1
            continue

        # 2. Lookback: find closest past date within LOOKBACK_DAYS
        d = pd.to_datetime(date)
        best = None
        for past in pd.date_range(end=d - pd.Timedelta(days=1), periods=LOOKBACK_DAYS, freq="D"):
            key = (tic, past.strftime("%Y-%m-%d"))
            if key in lookup:
                best = lookup[key]
                break

        if best:
            for col in SIGNAL_COLS:
                df.at[idx, col] = best[col]
            filled_lookback += 1
        else:
            kept_neutral += 1

    total = len(df)
    print(f"  Exact match   : {filled_exact:>6,} ({filled_exact/total*100:.1f}%)")
    print(f"  Lookback fill : {filled_lookback:>6,} ({filled_lookback/total*100:.1f}%)")
    print(f"  Kept neutral  : {kept_neutral:>6,} ({kept_neutral/total*100:.1f}%)")
    return df


def main():
    print("Loading scored news …")
    news   = pd.read_csv("multi_signal_nasdaq_news.csv")
    lookup = build_signal_lookup(news)

    for fname in [
        "train_data_multi_signal_2013_2018.csv",
        "trade_data_multi_signal_2019_2023.csv",
    ]:
        print(f"\nProcessing {fname} …")
        df = pd.read_csv(fname)
        df = fill_signals(df, lookup)
        df.to_csv(fname, index=False)
        scored_pct = (df["llm_sentiment"] != NEUTRAL).mean() * 100
        print(f"  Saved. Rows with real signals: {scored_pct:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
