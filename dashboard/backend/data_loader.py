"""
Data loader for the FinRL dashboard backend.

Reads CSV files produced by the training / backtesting pipeline and returns
structured data for the API.  Falls back to synthetic demo data when no real
CSV files are present so the frontend can still be developed / demoed.
"""

import os
import random
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Root directory: two levels up from this file
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _root(filename: str) -> str:
    return os.path.join(ROOT, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generators (used when real CSVs are absent)
# ─────────────────────────────────────────────────────────────────────────────

def _trading_dates(start: str = "2019-01-02", end: str = "2023-12-29") -> list[str]:
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _gbm_series(
    n: int,
    mu: float = 0.0003,
    sigma: float = 0.012,
    start: float = 1_000_000.0,
    seed: int = 42,
) -> list[float]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    vals = start * np.cumprod(1 + rets)
    return [start] + vals.tolist()


def _synthetic_portfolio() -> dict:
    dates = _trading_dates()
    n = len(dates)
    agents = {
        "PPO":             _gbm_series(n, mu=0.00028, sigma=0.013, seed=1),
        "CPPO":            _gbm_series(n, mu=0.00020, sigma=0.009, seed=2),
        "PPO-DeepSeek":    _gbm_series(n, mu=0.00035, sigma=0.014, seed=3),
        "CPPO-DeepSeek":   _gbm_series(n, mu=0.00030, sigma=0.010, seed=4),
        "CPPO-MultiSignal":_gbm_series(n, mu=0.00038, sigma=0.011, seed=5),
        "Regime-Switch":   _gbm_series(n, mu=0.00042, sigma=0.010, seed=6),
        "NASDAQ-100 (QQQ)":_gbm_series(n, mu=0.00033, sigma=0.015, seed=7),
    }
    # Trim all series to n+1 == len(dates)+1 using dates + one leading point
    return {"dates": dates, "agents": agents}


def _synthetic_regime(n: int) -> list[str]:
    """Alternating 30-day bull / bear blocks."""
    labels = []
    for i in range(n):
        labels.append("bear" if (i // 30) % 3 == 2 else "bull")
    return labels


def _synthetic_signals(n_dates: int = 20) -> list[dict]:
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]
    dates = _trading_dates()[-n_dates:]
    rng = np.random.default_rng(99)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "llm_sentiment": int(rng.integers(1, 6)),
                "llm_risk": int(rng.integers(1, 6)),
                "llm_confidence": int(rng.integers(2, 6)),
                "llm_volatility_forecast": int(rng.integers(1, 6)),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Real CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_portfolio_csv(filename: str) -> Optional[pd.DataFrame]:
    path = _root(filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "account_value" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _load_backtest_results() -> Optional[pd.DataFrame]:
    """Load real backtest results from backtest_results/portfolio_value.csv."""
    path = _root("backtest_results/portfolio_value.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "cppo_value" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _load_multi_signal_csv(filename: str) -> Optional[pd.DataFrame]:
    path = _root(filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_data() -> dict:
    """Return portfolio series for all available agents."""
    # Prefer real backtest data
    bt = _load_backtest_results()
    if bt is not None and len(bt) > 10:
        dates  = bt["date"].tolist()
        agents = {
            "CPPO-MultiSignal": bt["cppo_value"].tolist(),
            "Buy & Hold (EW)":  bt["bh_value"].tolist(),
        }
        # Normalise series lengths
        n = len(dates)
        for k in list(agents.keys()):
            s = agents[k]
            if len(s) > n:
                agents[k] = s[:n]
            elif len(s) < n:
                agents[k] = s + [s[-1]] * (n - len(s))
        return {"dates": dates, "agents": agents}

    # Fall back to synthetic demo data
    synth = _synthetic_portfolio()
    dates = synth["dates"]
    agents = dict(synth["agents"])

    # Try to overlay legacy CSV formats
    csv_map = {
        "Regime-Switch":    "regime_switch_portfolio.csv",
        "CPPO-MultiSignal": "results_cppo_multi_signal.csv",
    }
    for agent_name, filename in csv_map.items():
        df = _load_portfolio_csv(filename)
        if df is not None and len(df) > 10:
            agents[agent_name] = df["account_value"].tolist()
            if "date" in df.columns and len(df["date"]) == len(dates):
                dates = df["date"].tolist()

    # Normalise series lengths
    n = len(dates)
    for k in list(agents.keys()):
        s = agents[k]
        if len(s) > n + 1:
            agents[k] = s[:n + 1]
        elif len(s) < n + 1:
            agents[k] = s + [s[-1]] * (n + 1 - len(s))

    return {"dates": dates, "agents": agents}


def get_backtest_metrics() -> dict:
    """Return real backtest metrics from backtest_results/metrics.json."""
    import json
    path = _root("backtest_results/metrics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_regime_data() -> list[dict]:
    """Return list of {date, regime} dicts."""
    synth = _synthetic_portfolio()
    dates = synth["dates"]

    regime_csv = _load_portfolio_csv("regime_switch_portfolio.csv")
    if regime_csv is not None and "regime" in regime_csv.columns:
        regimes = regime_csv["regime"].tolist()
        csv_dates = regime_csv["date"].tolist() if "date" in regime_csv.columns else dates
        return [{"date": d, "regime": r} for d, r in zip(csv_dates, regimes)]

    regimes = _synthetic_regime(len(dates))
    return [{"date": d, "regime": r} for d, r in zip(dates, regimes)]


def get_llm_signals(limit: int = 200) -> list[dict]:
    """Return recent LLM signal rows."""
    sig_csv = _load_multi_signal_csv("multi_signal_nasdaq_news.csv")
    if sig_csv is not None:
        cols = ["Date", "Stock_symbol", "llm_sentiment", "llm_risk",
                "llm_confidence", "llm_volatility_forecast"]
        avail = [c for c in cols if c in sig_csv.columns]
        df = sig_csv[avail].dropna().tail(limit)
        df = df.rename(columns={"Date": "date", "Stock_symbol": "ticker"})
        return df.to_dict(orient="records")

    return _synthetic_signals(20)


def get_trade_data_signals(limit: int = 500) -> list[dict]:
    """Return LLM signals from the trade CSV (2019–2023)."""
    trade_csv = _load_multi_signal_csv("trade_data_multi_signal_2019_2023.csv")
    if trade_csv is not None:
        sig_cols = ["date", "tic", "llm_sentiment", "llm_risk",
                    "llm_confidence", "llm_volatility_forecast"]
        avail = [c for c in sig_cols if c in trade_csv.columns]
        if len(avail) >= 3:
            df = trade_csv[avail].dropna().tail(limit)
            df = df.rename(columns={"tic": "ticker"})
            return df.to_dict(orient="records")

    return _synthetic_signals(20)
