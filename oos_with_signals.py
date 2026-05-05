"""
OOS Evaluation with Real LLM Signals (2024-2025)

Merges the scored OOS news (from score_oos_news.py) with
the downloaded OHLCV data and re-runs the agent.

Compares:
  1. CPPO + neutral signals (3.0)     — baseline OOS
  2. CPPO + real LLM signals          — live LLM OOS
  3. Buy & Hold

If real signals outperform neutral, it confirms that LLM signals
add value even in a strict out-of-sample setting.

Usage:
    python oos_with_signals.py
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv

try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

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

SIGNAL_COLS = ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]


def merge_signals(oos_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge scored OOS signals into the price dataframe.
    For each (tic, date) try exact match, then 3-day lookback.
    Default to neutral (3.0) if no signal available.
    """
    df = oos_df.copy()
    for col in SIGNAL_COLS:
        df[col] = 3.0  # neutral default

    # Build lookup: (tic, date) → mean signals
    lookup = {}
    for _, row in signals_df.iterrows():
        key = (str(row["tic"]), str(row["date"]))
        lookup[key] = {col: float(row[col]) for col in SIGNAL_COLS if col in row}

    dates_list = sorted(df["date"].unique())
    date_set   = set(dates_list)

    for idx, row in df.iterrows():
        tic  = str(row["tic"])
        date = str(row["date"])

        # Exact match
        if (tic, date) in lookup:
            for col, val in lookup[(tic, date)].items():
                df.at[idx, col] = val
            continue

        # Lookback up to 3 days
        try:
            d_pos = dates_list.index(date)
            for back in range(1, 4):
                if d_pos - back >= 0:
                    prev_date = dates_list[d_pos - back]
                    if (tic, prev_date) in lookup:
                        for col, val in lookup[(tic, prev_date)].items():
                            df.at[idx, col] = val
                        break
        except (ValueError, IndexError):
            pass

    return df


def run_agent(model, df, unique_dates):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--signals",    default="oos_signals_2024_2025.csv")
    parser.add_argument("--oos_data",   default="backtest_results/oos_portfolio.csv")
    parser.add_argument("--start",      default="2024-01-01")
    parser.add_argument("--end",        default="2025-12-31")
    args = parser.parse_args()

    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("paper/figures",    exist_ok=True)

    if not os.path.exists(args.signals):
        print(f"Signals file not found: {args.signals}")
        print("Run: python score_oos_news.py")
        return

    print("Loading OOS signals …")
    signals_df = pd.read_csv(args.signals)
    print(f"  {len(signals_df)} (tic, date) entries")
    print(f"  Coverage: {(signals_df['llm_sentiment'] != 3.0).mean()*100:.1f}% non-neutral")

    print("Downloading OOS OHLCV data …")
    from oos_evaluation import (
        download_oos_data, add_technical_indicators,
        add_turbulence, prepare_for_env, bh_series,
    )
    NASDAQ_30 = signals_df["tic"].unique().tolist()[:30]
    df_raw    = download_oos_data(NASDAQ_30, args.start, args.end)
    if df_raw is None:
        return

    df_feat = add_technical_indicators(df_raw)
    df_feat = add_turbulence(df_feat)

    # Two versions: neutral signals vs real signals
    df_neutral = df_feat.copy()
    for col in SIGNAL_COLS:
        df_neutral[col] = 3.0

    df_scored = merge_signals(df_feat, signals_df)

    # Coverage report
    non_neutral = (df_scored["llm_sentiment"] != 3.0).sum()
    total       = len(df_scored)
    print(f"  After merge: {non_neutral/total*100:.1f}% non-neutral signals")

    # Prepare environments
    df_neutral_env, unique_dates = prepare_for_env(df_neutral)
    df_scored_env, _             = prepare_for_env(df_scored)
    n = len(unique_dates)

    bh = bh_series(df_neutral_env, n)

    print(f"\nLoading model: {args.model}")
    model = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    print("Running CPPO + neutral signals (OOS baseline) …")
    pvs_neutral = run_agent(model, df_neutral_env, unique_dates)

    print("Running CPPO + real LLM signals (OOS + live scoring) …")
    pvs_scored  = run_agent(model, df_scored_env, unique_dates)

    m_neutral = compute_full_metrics(pvs_neutral, bh, name="CPPO (OOS, neutral)")
    m_scored  = compute_full_metrics(pvs_scored,  bh, name="CPPO (OOS, LLM signals)")
    m_bh      = compute_full_metrics(bh,           bh, name="Buy & Hold (EW)")

    print_metrics(m_scored)
    print_metrics(m_neutral)

    print(f"\n{'─'*55}")
    print("  LLM Signal Value in OOS Period")
    print(f"{'─'*55}")
    print(f"  CPPO + LLM signals  : CR={m_scored['cumulative_return']:.2f}%  "
          f"Sharpe={m_scored['sharpe_ratio']:.4f}")
    print(f"  CPPO + neutral      : CR={m_neutral['cumulative_return']:.2f}%  "
          f"Sharpe={m_neutral['sharpe_ratio']:.4f}")
    print(f"  Signal contribution : ΔCR={m_scored['cumulative_return']-m_neutral['cumulative_return']:+.2f}pp  "
          f"ΔSharpe={m_scored['sharpe_ratio']-m_neutral['sharpe_ratio']:+.4f}")

    # Save
    pd.DataFrame({
        "date":             unique_dates[:n],
        "cppo_llm_signals": pvs_scored[:n],
        "cppo_neutral":     pvs_neutral[:n],
        "buy_hold":         bh[:n],
    }).to_csv("backtest_results/oos_signal_comparison.csv", index=False)

    # Plot
    dates_dt = pd.to_datetime(unique_dates[:n])
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.plot(dates_dt, np.array(pvs_scored)/pvs_scored[0],
            color="#2563EB", lw=2.2, label="CPPO + LLM signals (live)")
    ax.plot(dates_dt, np.array(pvs_neutral)/pvs_neutral[0],
            color="#F59E0B", lw=1.8, ls="--", label="CPPO + neutral signals")
    ax.plot(dates_dt, bh[:n]/bh[0],
            color="#94A3B8", lw=1.5, ls=":", label="Buy & Hold (EW)")
    ax.set_title("OOS 2024-2025: LLM Signals vs Neutral vs Buy & Hold",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value (start=1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig("paper/figures/fig_oos_signal_value.png", dpi=150)
    plt.close(fig)
    print("\nSaved → paper/figures/fig_oos_signal_value.png")


if __name__ == "__main__":
    main()
