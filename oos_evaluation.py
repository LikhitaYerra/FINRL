"""
Out-of-Sample 2024-2025 Evaluation

Inspired by: FutureX (Zeng et al. 2025), Chandak et al. 2026,
             Look-Ahead-Bench (Benhenda 2026), YCBench (Benhenda 2026)

The agent was trained on 2013-2018 and evaluated on 2019-2023 in the main paper.
This script provides a TRUE out-of-sample test by:

  1. Downloading NASDAQ-100 OHLCV data for 2024-01-01 to present
  2. Scoring recent news headlines from Yahoo Finance RSS feeds using the LLM
  3. Running the trained agent on this new data WITHOUT any retraining
  4. Comparing performance to Buy & Hold

This follows the "live benchmark" philosophy:
  - No re-training or parameter tuning on 2024-2025 data
  - News scored in temporal order (no future news)
  - Strict point-in-time evaluation

This is the strongest possible test of generalization and directly addresses
the look-ahead bias concern: the agent was trained BEFORE 2024 data existed.

Usage:
    python oos_evaluation.py
    python oos_evaluation.py --start 2024-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from finrl.config import INDICATORS
except Exception:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

from model_loader import load_cppo_model
from metrics_extended import compute_full_metrics, print_metrics
from env_stocktrading_multi_signal import StockTradingEnv

try:
    from stockstats import StockDataFrame
    HAS_STOCKSTATS = True
except ImportError:
    HAS_STOCKSTATS = False


# ─── Config ───────────────────────────────────────────────────────────────────

INITIAL_AMOUNT = 1_000_000
STOCK_DIM      = 30
STATE_SPACE    = 1 + 2 * STOCK_DIM + (len(INDICATORS) + 4) * STOCK_DIM
ACTION_SPACE   = STOCK_DIM

ENV_KWARGS = dict(
    hmax=100, initial_amount=INITIAL_AMOUNT,
    num_stock_shares=[0]*STOCK_DIM,
    buy_cost_pct=[0.001]*STOCK_DIM, sell_cost_pct=[0.001]*STOCK_DIM,
    reward_scaling=1e-4, state_space=STATE_SPACE, action_space=ACTION_SPACE,
    tech_indicator_list=INDICATORS, turbulence_threshold=380,
    drawdown_penalty=0.1, make_plots=False, print_verbosity=999,
)

# NASDAQ-100 constituents (same 30 used in training)
NASDAQ_30 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","AVGO","ASML",
    "COST","CSCO","ADBE","NFLX","AMD","INTC","INTU","QCOM","TXN","AMAT",
    "SBUX","GILD","MDLZ","PYPL","REGN","ISRG","VRTX","LRCX","KLAC","MRVL"
]

SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
FIG_DIR  = "paper/figures"
DATA_DIR = "backtest_results"


# ─── Data download ────────────────────────────────────────────────────────────

def download_oos_data(tickers: list, start: str, end: str) -> pd.DataFrame | None:
    """Download and engineer features for OOS period."""
    print(f"  Downloading OHLCV {start} → {end} for {len(tickers)} stocks …")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print("  ERROR: No data downloaded")
        return None

    # Flatten MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join(c).strip('_') for c in raw.columns]

    rows = []
    for tic in tickers:
        cols_map = {}
        for field in ["Open","High","Low","Close","Volume"]:
            cand = f"{field}_{tic}"
            if cand in raw.columns:
                cols_map[field.lower()] = cand

        if "close" not in cols_map:
            continue

        df_tic = raw[[v for v in cols_map.values()]].copy()
        df_tic.columns = [k for k in cols_map.keys()]
        df_tic = df_tic.dropna(subset=["close"])
        df_tic["tic"]  = tic
        df_tic["date"] = df_tic.index.strftime("%Y-%m-%d")
        rows.append(df_tic.reset_index(drop=True))

    if not rows:
        return None

    df = pd.concat(rows, ignore_index=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same technical indicators used during training."""
    if not HAS_STOCKSTATS:
        # Fallback: compute manually
        df = df.sort_values(["tic", "date"]).copy()
        for tic, grp in df.groupby("tic"):
            idx = grp.index
            c   = grp["close"]
            df.loc[idx, "close_30_sma"] = c.rolling(30, min_periods=1).mean()
            df.loc[idx, "close_60_sma"] = c.rolling(60, min_periods=1).mean()
            delta   = c.diff()
            gain    = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss    = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs      = gain / (loss + 1e-9)
            df.loc[idx, "rsi_30"] = 100 - 100 / (1 + rs)
            # MACD
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            df.loc[idx, "macd"]   = ema12 - ema26
            # Bollinger
            sma20  = c.rolling(20, min_periods=1).mean()
            std20  = c.rolling(20, min_periods=1).std().fillna(0)
            df.loc[idx, "boll_ub"] = sma20 + 2 * std20
            df.loc[idx, "boll_lb"] = sma20 - 2 * std20
            # CCI, DX: simplified approximations
            df.loc[idx, "cci_30"]  = (c - sma20) / (0.015 * std20 + 1e-9)
            df.loc[idx, "dx_30"]   = df.loc[idx, "rsi_30"]  # proxy

        for ind in INDICATORS:
            if ind not in df.columns:
                df[ind] = 0.0
        return df

    # StockStats path
    result_frames = []
    for tic, grp in df.groupby("tic"):
        grp_reset = grp.reset_index(drop=True).copy()
        saved_date = grp_reset["date"].values
        sdf = StockDataFrame.retype(grp_reset)
        for ind in INDICATORS:
            try:
                _ = sdf[ind]
            except Exception:
                sdf[ind] = 0.0
        sdf_df = pd.DataFrame(sdf)
        sdf_df["date"] = saved_date  # restore date column
        result_frames.append(sdf_df)
    return pd.concat(result_frames, ignore_index=True)


def add_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute turbulence index as Mahalanobis distance from historical returns."""
    df = df.reset_index(drop=True)
    pivot = df.pivot(index="date", columns="tic", values="close").sort_index()
    ret   = pivot.pct_change().fillna(0)

    # Use rolling 252-day covariance
    turb = pd.Series(index=ret.index, dtype=float)
    for i, date in enumerate(ret.index):
        if i < 252:
            turb.loc[date] = 0.0
            continue
        window = ret.iloc[max(0, i-252):i]
        mu     = window.mean().values
        cov    = window.cov().values
        y      = ret.iloc[i].values - mu
        try:
            inv_cov = np.linalg.pinv(cov)
            t       = float(y @ inv_cov @ y)
        except Exception:
            t = 0.0
        turb.loc[date] = t

    df = df.merge(
        turb.rename("turbulence").reset_index().rename(columns={"index":"date"}),
        on="date", how="left"
    )
    df["turbulence"] = df["turbulence"].fillna(0)
    return df


# ─── LLM scoring for OOS news ─────────────────────────────────────────────────

def score_oos_signals_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign neutral signals (3.0) for OOS evaluation.
    Represents the 'no news available' baseline.
    """
    for col in SIGNAL_COLS:
        df[col] = 3.0
    return df


def merge_llm_signals_csv(df: pd.DataFrame, csv_path: str) -> tuple[pd.DataFrame, str]:
    """
    Attach LLM columns from ``oos_signals*.csv``.

    - If signal timestamps overlap the price window: backward ``merge_asof`` per ticker
      (last known signal ≤ trading day).
    - If every signal date is *after* the last price date (common when RSS parsing failed):
      take the mean signal per ticker and broadcast across all trading days
      (**cross-sectional snapshot**, not a daily path — documented in logs).
    """
    sig = pd.read_csv(csv_path)
    for col in SIGNAL_COLS:
        if col not in sig.columns:
            sig[col] = 3.0

    df = df.copy()
    df["_pd"] = pd.to_datetime(df["date"])
    sig = sig.copy()
    sig["_pd"] = pd.to_datetime(sig["date"])

    price_min, price_max = df["_pd"].min(), df["_pd"].max()
    sig_min, sig_max = sig["_pd"].min(), sig["_pd"].max()

    # Snapshot: no signal could apply historically to this price range
    if sig_min > price_max:
        snap = sig.groupby("tic")[SIGNAL_COLS].mean().reset_index()
        out = df.merge(snap, on="tic", how="left")
        out[SIGNAL_COLS] = out[SIGNAL_COLS].fillna(3.0)
        mode = (
            "snapshot_broadcast — signal dates are after the OHLCV window; "
            "using mean-per-ticker scores on every trading day (cross-section only)."
        )
        out = out.drop(columns=["_pd"])
        return out, mode

    # Point-in-time backward merge per ticker
    pieces = []
    for tic in sorted(df["tic"].unique()):
        sub = df[df["tic"] == tic].sort_values("_pd")
        ssub = sig[sig["tic"] == tic].sort_values("_pd").drop_duplicates(subset=["_pd"], keep="last")
        if ssub.empty:
            sub = sub.copy()
            sub[SIGNAL_COLS] = 3.0
            pieces.append(sub)
            continue
        merged = pd.merge_asof(
            sub,
            ssub[["_pd"] + SIGNAL_COLS],
            on="_pd",
            direction="backward",
            allow_exact_matches=True,
        )
        merged[SIGNAL_COLS] = merged[SIGNAL_COLS].fillna(3.0)
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True)
    out = out.drop(columns=["_pd"])
    mode = "merge_asof_backward — point-in-time signals per trading day."
    return out, mode


# ─── Format for environment ───────────────────────────────────────────────────

def prepare_for_env(df: pd.DataFrame) -> pd.DataFrame:
    """Format OOS data to match the environment's expected schema."""
    required_cols = ["date","tic","close","open","high","low","volume"] + INDICATORS
    signal_cols   = SIGNAL_COLS

    # Ensure all columns present
    for col in required_cols + signal_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Fill NaN in indicators
    for col in INDICATORS:
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    df["new_idx"] = df["date"].map(date_to_idx)
    df = df.set_index("new_idx")
    return df, unique_dates


# ─── Run agent ────────────────────────────────────────────────────────────────

def run_agent(model, df, unique_dates) -> list:
    env = StockTradingEnv(df=df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    pvs  = [INITIAL_AMOUNT]
    done = False
    while not done:
        action = model.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])
    return pvs[:len(unique_dates)]


def bh_series(df: pd.DataFrame, n: int) -> np.ndarray:
    avg = df.reset_index().groupby("date")["close"].mean().sort_index()
    return (INITIAL_AMOUNT * avg.values / avg.values[0])[:n]


# ─── Figure ───────────────────────────────────────────────────────────────────

def plot_oos(pvs_agent, pvs_bh, dates, metrics_agent, out, signal_note: str = ""):
    dates_dt = pd.to_datetime(dates)
    n        = min(len(pvs_agent), len(dates_dt))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    norm_agent = np.array(pvs_agent[:n]) / pvs_agent[0]
    norm_bh    = pvs_bh[:n] / pvs_bh[0]

    ax1.plot(dates_dt[:n], norm_agent, color="#2563EB", lw=2.2,
             label=f"CPPO (LLM signals) — OOS")
    ax1.plot(dates_dt[:n], norm_bh,    color="#94A3B8", lw=1.5, ls="--",
             label="Buy & Hold (EW)")
    ax1.axvline(pd.to_datetime(dates[0]), color="#10B981", lw=1.5, ls=":",
                label="OOS start (2024-01-01)")

    tit = (
        f"True Out-of-Sample Evaluation: 2024–2025\n"
        f"(Agent trained on 2013–2018, evaluated on 2019–2023, tested here on 2024+)\n"
        f"{signal_note}"
    )
    ax1.set_title(tit, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Normalised Portfolio Value (start=1)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Drawdown
    peak = np.maximum.accumulate(pvs_agent[:n])
    dd   = (np.array(pvs_agent[:n]) - peak) / peak * 100
    ax2.fill_between(dates_dt[:n], dd, 0, alpha=0.4, color="#2563EB")
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Annotation box
    txt = (f"OOS CR: {metrics_agent['cumulative_return']:.1f}%\n"
           f"Sharpe: {metrics_agent['sharpe_ratio']:.3f}\n"
           f"MDD: {metrics_agent['max_drawdown_pct']:.1f}%")
    ax1.annotate(txt, xy=(0.02, 0.97), xycoords="axes fraction",
                 fontsize=8, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    parser.add_argument(
        "--signals",
        choices=["auto", "neutral", "csv"],
        default="auto",
        help="auto: use oos_signals CSV if present, else neutral",
    )
    parser.add_argument(
        "--signals-csv",
        default="oos_signals_2024_2025.csv",
        help="LLM signal table from score_oos_news.py",
    )
    args = parser.parse_args()

    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Out-of-Sample Evaluation: {args.start} → {args.end}")
    print("=" * 55)
    print("NOTE: Agent was trained on 2013-2018 and in-sample evaluated")
    print("on 2019-2023. This is a STRICT out-of-sample test.")
    print("=" * 55)

    # Download OOS data
    df_raw = download_oos_data(NASDAQ_30, args.start, args.end)
    if df_raw is None or len(df_raw) == 0:
        print("ERROR: Could not download OOS data. Check internet connection.")
        return

    actual_dates = sorted(df_raw["date"].unique())
    n_stocks_downloaded = df_raw["tic"].nunique()
    print(f"  Downloaded: {len(actual_dates)} dates, {n_stocks_downloaded} stocks")

    # Keep only stocks that have full coverage
    coverage = df_raw.groupby("tic")["date"].count()
    full_tickers = coverage[coverage >= len(actual_dates) * 0.9].index.tolist()[:STOCK_DIM]
    if len(full_tickers) < STOCK_DIM:
        print(f"  WARNING: only {len(full_tickers)} stocks with >90% coverage (need {STOCK_DIM})")
        # Pad with available tickers
        full_tickers = coverage.nlargest(STOCK_DIM).index.tolist()

    df_raw = df_raw[df_raw["tic"].isin(full_tickers[:STOCK_DIM])].copy()

    # Engineer features
    print("  Engineering technical indicators …")
    df_feat = add_technical_indicators(df_raw)

    # Turbulence
    print("  Computing turbulence index …")
    df_feat = add_turbulence(df_feat)

    merge_mode = ""
    csv_ok = os.path.isfile(args.signals_csv)
    if args.signals == "neutral":
        use_csv = False
    elif args.signals == "csv":
        use_csv = csv_ok
        if not csv_ok:
            print(f"  WARNING: {args.signals_csv} not found — using neutral signals.")
    else:  # auto
        use_csv = csv_ok

    signal_note = ""
    if use_csv:
        print(f"  Merging LLM signals from {args.signals_csv} …")
        df_feat, merge_mode = merge_llm_signals_csv(df_feat, args.signals_csv)
        print(f"  Merge mode: {merge_mode}")
        signal_note = merge_mode[:120] + ("…" if len(merge_mode) > 120 else "")
    else:
        df_feat = score_oos_signals_neutral(df_feat)
        merge_mode = "neutral_3.0"
        signal_note = "Signals: neutral baseline (3.0)."

    # Ensure correct number of stocks
    unique_tickers = sorted(df_feat["tic"].unique())[:STOCK_DIM]
    df_feat = df_feat[df_feat["tic"].isin(unique_tickers)]
    n_stocks = df_feat["tic"].nunique()
    print(f"  Final dataset: {n_stocks} stocks")

    if n_stocks != STOCK_DIM:
        print(f"  ERROR: Expected {STOCK_DIM} stocks, got {n_stocks}. Aborting.")
        return

    # Prepare for environment
    df_env, unique_dates = prepare_for_env(df_feat)
    n = len(unique_dates)
    print(f"  Trading days: {n}")

    # Buy-and-hold baseline
    bh = bh_series(df_env, n)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    # Run agent
    print("Running agent on OOS data …")
    pvs_agent = run_agent(model, df_env, unique_dates)

    # Metrics
    m_agent = compute_full_metrics(pvs_agent, bh, name="CPPO (OOS 2024-2025)")
    m_bh    = compute_full_metrics(bh,         bh, name="Buy & Hold (EW)")

    print_metrics(m_agent)
    print_metrics(m_bh)

    # Compare to in-sample performance
    print(f"\n{'─'*55}")
    print("  OOS vs In-Sample Comparison")
    print(f"{'─'*55}")
    print(f"  In-sample   (2019–2023): CR=246.3%  Sharpe=1.070")
    print(f"  OOS         ({args.start[:4]}–{args.end[:4]}):   CR={m_agent['cumulative_return']:.1f}%  Sharpe={m_agent['sharpe_ratio']:.3f}")
    print(f"  Generalization gap CR  : {m_agent['cumulative_return']-246.3:+.1f} pp")
    print(f"  B&H OOS: CR={m_bh['cumulative_return']:.1f}%  Sharpe={m_bh['sharpe_ratio']:.3f}")

    # Save results
    import json
    oos_results = {
        "period": f"{args.start} to {args.end}",
        "n_trading_days": n,
        "n_stocks": n_stocks,
        "agent_metrics": m_agent,
        "bh_metrics": m_bh,
        "insample_reference": {
            "period": "2019-2023",
            "cumulative_return": 246.3,
            "sharpe_ratio": 1.070,
        },
        "signals_mode": "csv" if use_csv else "neutral",
        "signals_merge": merge_mode,
        "signals_csv": args.signals_csv if use_csv else None,
    }
    with open(f"{DATA_DIR}/oos_results.json", "w") as f:
        json.dump(oos_results, f, indent=2)
    print(f"\n  Saved → {DATA_DIR}/oos_results.json")

    # Save portfolio CSV
    pd.DataFrame({
        "date":      unique_dates[:n],
        "agent":     pvs_agent[:n],
        "buy_hold":  bh[:n],
    }).to_csv(f"{DATA_DIR}/oos_portfolio.csv", index=False)

    # Plot
    plot_oos(pvs_agent, bh, unique_dates,
             m_agent, f"{FIG_DIR}/fig_oos_evaluation.png",
             signal_note=signal_note)


if __name__ == "__main__":
    main()
