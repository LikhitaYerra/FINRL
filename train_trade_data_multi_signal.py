"""
Data preparation for FinRL Contest 2025 Task 1 — multi-signal variant.

Merges 4 LLM signals (sentiment, risk, confidence, volatility_forecast)
produced by score_multi_signal.py with NASDAQ-100 OHLCV + technical indicators.

Outputs:
  train_data_multi_signal_2013_2018.csv
  trade_data_multi_signal_2019_2023.csv

Usage:
    python train_trade_data_multi_signal.py \
        --signals multi_signal_nasdaq_news.csv

Or load pre-scored data from Hugging Face:
    python train_trade_data_multi_signal.py --hf_dataset benstaf/multi_signal_nasdaq
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TRAIN_START_DATE = "2013-01-01"
TRAIN_END_DATE = "2018-12-31"
TRADE_START_DATE = "2019-01-01"
TRADE_END_DATE = "2023-12-31"

NASDAQ_100_TICKERS = [
    "ADBE", "ADP", "ABNB", "ALGN", "GOOGL", "GOOG", "AMZN", "AMD", "AEP", "AMGN",
    "ADI", "ANSS", "AAPL", "AMAT", "ASML", "AZN", "TEAM", "ADSK", "BKR", "BIIB",
    "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CTSH", "CMCSA", "CEG", "CPRT",
    "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DLTR", "EBAY", "EA",
    "ENPH", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "ILMN",
    "INTC", "INTU", "ISRG", "JD", "KDP", "KLAC", "KHC", "LRCX", "LCID", "LULU",
    "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT", "MRNA", "MDLZ", "MNST",
    "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PANW", "PAYX", "PYPL",
    "PDD", "PEP", "QCOM", "REGN", "ROST", "SGEN", "SIRI", "SBUX", "SNPS", "TMUS",
    "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS",
]

SIGNAL_COLS = [
    "llm_sentiment",
    "llm_risk",
    "llm_confidence",
    "llm_volatility_forecast",
]

# Default fill values when a signal is missing for a given (date, ticker)
SIGNAL_DEFAULTS = {
    "llm_sentiment": 3.0,        # neutral
    "llm_risk": 3.0,             # neutral
    "llm_confidence": 3.0,       # neutral
    "llm_volatility_forecast": 3.0,  # neutral
}


# --------------------------------------------------------------------------- #
# Price data helpers
# --------------------------------------------------------------------------- #

def fetch_price_data(tickers: list[str]) -> pd.DataFrame:
    """Download OHLCV + indicators for all tickers, handling SGEN manually."""
    position = tickers.index("SGEN")
    tickers_no_sgen = [t for t in tickers if t != "SGEN"]

    df_raw = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=tickers_no_sgen,
    ).fetch_data()

    # Re-insert SGEN if local CSV is available
    if os.path.exists("seagen.csv"):
        df_seagen = _load_sgen()
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = pd.concat([df_raw, df_seagen], ignore_index=True)
        df_raw.sort_values("date", inplace=True)
        df_raw["date"] = df_raw["date"].dt.strftime("%Y-%m-%d")

    tickers_no_sgen.insert(position, "SGEN")
    return df_raw


def _load_sgen() -> pd.DataFrame:
    df = pd.read_csv("seagen.csv")
    df.rename(
        columns={
            "Date": "date",
            "Price": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Vol.": "volume",
            "Change %": "change_percent",
        },
        inplace=True,
    )
    df["tic"] = "SGEN"
    df["volume"] = df["volume"].apply(_parse_volume)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df.drop(columns=["change_percent"], errors="ignore", inplace=True)
    return df


def _parse_volume(vol_str) -> float:
    if isinstance(vol_str, str):
        if vol_str.endswith("M"):
            return float(vol_str.replace("M", "")) * 1e6
        if vol_str.endswith("K"):
            return float(vol_str.replace("K", "")) * 1e3
    return float(vol_str)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))
    processed_full = (
        pd.DataFrame(combination, columns=["date", "tic"])
        .merge(processed, on=["date", "tic"], how="left")
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.ffill()
    return processed_full


# --------------------------------------------------------------------------- #
# Signal merging
# --------------------------------------------------------------------------- #

def load_signals(signals_path: str) -> pd.DataFrame:
    """Load the multi-signal CSV produced by score_multi_signal.py."""
    sig = pd.read_csv(signals_path, on_bad_lines="warn", engine="python")
    sig.columns = sig.columns.str.capitalize()

    # Normalise column names
    rename_map = {}
    for col in sig.columns:
        cl = col.lower()
        if "sentiment" in cl:
            rename_map[col] = "llm_sentiment"
        elif "risk" in cl:
            rename_map[col] = "llm_risk"
        elif "confidence" in cl:
            rename_map[col] = "llm_confidence"
        elif "volatility" in cl:
            rename_map[col] = "llm_volatility_forecast"
        elif col.lower() == "stock_symbol":
            rename_map[col] = "tic"
        elif col.lower() == "date":
            rename_map[col] = "Date"
    sig.rename(columns=rename_map, inplace=True)

    sig["Date"] = pd.to_datetime(sig["Date"]).dt.tz_localize(None)
    return sig


def merge_signals(price_df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Left-join all four LLM signals onto the price dataframe."""
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    avail_signal_cols = [c for c in SIGNAL_COLS if c in signals.columns]
    merge_cols = ["Date", "tic"] + avail_signal_cols

    merged = df.merge(
        signals[merge_cols],
        left_on=["date", "tic"],
        right_on=["Date", "tic"],
        how="left",
    )
    merged.drop(columns=["Date"], errors="ignore", inplace=True)

    # Fill missing signals with neutral defaults
    for col in SIGNAL_COLS:
        if col not in merged.columns:
            merged[col] = SIGNAL_DEFAULTS[col]
        else:
            merged[col].fillna(SIGNAL_DEFAULTS[col], inplace=True)

    return merged


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals",
        default="multi_signal_nasdaq_news.csv",
        help="CSV output from score_multi_signal.py",
    )
    parser.add_argument(
        "--hf_dataset",
        default=None,
        help="Hugging Face dataset id to load signals from (optional)",
    )
    parser.add_argument(
        "--train_out", default="train_data_multi_signal_2013_2018.csv"
    )
    parser.add_argument(
        "--trade_out", default="trade_data_multi_signal_2019_2023.csv"
    )
    args = parser.parse_args()

    # ---- Load signals ----
    if args.hf_dataset:
        from datasets import load_dataset

        print(f"Loading signals from HuggingFace: {args.hf_dataset}")
        ds = load_dataset(args.hf_dataset)
        signals = pd.DataFrame(ds["train"])
    else:
        print(f"Loading signals from {args.signals}")
        signals = load_signals(args.signals)

    # ---- Fetch & engineer price data ----
    print("Fetching price data from Yahoo Finance …")
    df_raw = fetch_price_data(NASDAQ_100_TICKERS)

    print("Engineering features …")
    processed = engineer_features(df_raw)

    # ---- Split ----
    train = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed, TRADE_START_DATE, TRADE_END_DATE)

    # ---- Merge LLM signals ----
    print("Merging multi-signal LLM scores …")
    train_ms = merge_signals(train, signals)
    trade_ms = merge_signals(trade, signals)

    # ---- Validate ----
    for col in SIGNAL_COLS:
        nan_pct = train_ms[col].isna().mean() * 100
        print(f"  {col}: {nan_pct:.1f}% NaN after merge (filled with neutral)")

    # ---- Save ----
    train_ms.to_csv(args.train_out)
    trade_ms.to_csv(args.trade_out)
    print(f"Saved:\n  {args.train_out}\n  {args.trade_out}")
    print(f"Train shape: {train_ms.shape}  |  Trade shape: {trade_ms.shape}")
    print(f"Columns: {list(train_ms.columns)}")


if __name__ == "__main__":
    main()
