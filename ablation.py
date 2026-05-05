"""
Signal ablation study + baseline strategies.

For each ablation, we zero-out one LLM signal dimension and run the
trained model to measure the performance drop. This shows WHICH signals
carry the most alpha — a critical result for the paper.

Also implements non-RL baselines:
  - Momentum (buy top-K performers of last 20 days)
  - Equal-volatility weighting (risk parity)
  - Equal-weight buy-and-hold
  - Turbulence-filtered buy-and-hold

Usage:
    python ablation.py
    python ablation.py --model trained_models/agent_cppo_multi_signal_30_epochs.pth
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from metrics_extended import compute_full_metrics
from model_loader import load_cppo_model, MLPActorCritic
from env_stocktrading_multi_signal import StockTradingEnv
from finrl.config import INDICATORS


# ─── Config ──────────────────────────────────────────────────────────────────

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

SIGNAL_NAMES = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
NEUTRAL      = 3.0




# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_agent(ac, trade_df, unique_dates, signal_mask=None):
    """
    Run the RL agent. signal_mask: list of signal names to zero-out (ablation).
    """
    df = trade_df.copy()
    if signal_mask:
        for col in signal_mask:
            df[col] = NEUTRAL

    env = StockTradingEnv(df=df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    pvs  = [INITIAL_AMOUNT]
    done = False
    while not done:
        action = ac.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])

    return pvs[:len(unique_dates)]


def bh_equal_weight(trade_df, n):
    avg = trade_df.groupby("date")["close"].mean().sort_index()
    return (INITIAL_AMOUNT * avg.values / avg.values[0])[:n]


def momentum_strategy(trade_df, n_days_lookback=20, top_k=10, rebalance_freq=5):
    """
    Each rebalance_freq days, buy equal weight of top-K stocks by
    past n_days_lookback return. Simple transaction cost of 0.1%.
    """
    dates  = sorted(trade_df["date"].unique())
    pivoted = trade_df.pivot(index="date", columns="tic", values="close")
    pivoted = pivoted.sort_index()

    capital = INITIAL_AMOUNT
    holdings = {}  # tic -> shares
    pvs = [capital]

    for i, date in enumerate(dates[1:], 1):
        # Update portfolio value
        prices = pivoted.loc[date]
        pv = sum(holdings.get(t, 0) * p for t, p in prices.items()) + capital
        pvs.append(pv)

        # Rebalance
        if i % rebalance_freq == 0 and i >= n_days_lookback:
            # Compute past returns
            past_date = dates[max(0, i - n_days_lookback)]
            past_prices = pivoted.loc[past_date]
            returns = (prices - past_prices) / (past_prices + 1e-9)
            top_tickers = returns.nlargest(top_k).index.tolist()

            # Sell everything
            for t, sh in list(holdings.items()):
                p = prices.get(t, 0)
                capital += sh * p * (1 - 0.001)  # sell cost
            holdings = {}

            # Buy top-K equally
            alloc = pv / top_k
            for t in top_tickers:
                p = prices.get(t, 1)
                if p > 0:
                    shares = (alloc * (1 - 0.001)) / p
                    holdings[t]  = shares
                    capital -= alloc

    return np.array(pvs[:len(dates)])


def equal_volatility_strategy(trade_df, n):
    """
    Risk-parity: weight each stock inversely proportional to its 30-day vol.
    Rebalance monthly.
    """
    dates   = sorted(trade_df["date"].unique())
    pivoted = trade_df.pivot(index="date", columns="tic", values="close").sort_index()
    ret     = pivoted.pct_change()

    capital  = INITIAL_AMOUNT
    holdings = {}
    pvs      = [capital]
    WINDOW   = 30

    for i, date in enumerate(dates[1:], 1):
        prices = pivoted.loc[date]
        pv = sum(holdings.get(t, 0) * p for t, p in prices.items()) + capital
        pvs.append(pv)

        if i % 21 == 0 and i >= WINDOW:  # monthly rebalance
            # Compute vol weights
            past_ret = ret.iloc[max(0, i-WINDOW):i]
            vols     = past_ret.std().fillna(0.01)
            inv_vol  = 1.0 / (vols + 1e-9)
            weights  = inv_vol / inv_vol.sum()

            # Sell all
            for t, sh in list(holdings.items()):
                capital += sh * prices.get(t, 0) * 0.999
            holdings = {}

            # Buy
            for t, w in weights.items():
                p = prices.get(t, 1)
                if p > 0:
                    shares      = (pv * w * 0.999) / p
                    holdings[t] = shares
                    capital    -= pv * w

    return np.array(pvs[:n])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    args = parser.parse_args()

    os.makedirs("backtest_results", exist_ok=True)

    print("Loading data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    for col in SIGNAL_NAMES:
        trade[col] = trade.get(col, NEUTRAL).fillna(NEUTRAL)

    n = len(unique_dates)
    bh = bh_equal_weight(trade, n)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    ac = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    results = {}

    # ── Full model (baseline) ─────────────────────────────────────────────────
    print("\n[1/N] Full model (all signals) …")
    pvs = run_agent(ac, trade, unique_dates)
    results["CPPO (all signals)"] = pvs
    m = compute_full_metrics(pvs, bh, name="CPPO (all signals)")
    print(f"  CR={m['cumulative_return']:.2f}%  SR={m['sharpe_ratio']:.4f}  Rachev={m['rachev_ratio']:.4f}")

    # ── Ablations: remove each signal one at a time ───────────────────────────
    for sig in SIGNAL_NAMES:
        label = f"CPPO (no {sig.replace('llm_','')})"
        print(f"  Ablation: {label} …")
        pvs_abl = run_agent(ac, trade, unique_dates, signal_mask=[sig])
        results[label] = pvs_abl
        m_abl = compute_full_metrics(pvs_abl, bh, name=label)
        delta_cr = m_abl["cumulative_return"] - results["CPPO (all signals)"]
        delta_cr_vs_full = compute_full_metrics(pvs, bh)["cumulative_return"] - m_abl["cumulative_return"]
        print(f"  CR={m_abl['cumulative_return']:.2f}%  SR={m_abl['sharpe_ratio']:.4f}  "
              f"ΔCR_vs_full={-delta_cr_vs_full:+.2f}pp")

    # ── All signals neutral ────────────────────────────────────────────────────
    print("\n  Ablation: all signals neutral …")
    pvs_neutral = run_agent(ac, trade, unique_dates, signal_mask=SIGNAL_NAMES)
    results["CPPO (neutral)"] = pvs_neutral
    m_neutral = compute_full_metrics(pvs_neutral, bh, name="CPPO (neutral)")
    print(f"  CR={m_neutral['cumulative_return']:.2f}%  SR={m_neutral['sharpe_ratio']:.4f}")

    # ── Non-RL baselines ──────────────────────────────────────────────────────
    print("\n[2/N] Momentum strategy …")
    pvs_mom = momentum_strategy(trade, n_days_lookback=20, top_k=10)
    results["Momentum (top-10)"] = pvs_mom.tolist()
    m_mom = compute_full_metrics(pvs_mom, bh, name="Momentum (top-10)")
    print(f"  CR={m_mom['cumulative_return']:.2f}%  SR={m_mom['sharpe_ratio']:.4f}")

    print("[3/N] Equal-volatility strategy …")
    pvs_ev = equal_volatility_strategy(trade, n)
    results["Equal-Vol (risk parity)"] = pvs_ev.tolist()
    m_ev = compute_full_metrics(pvs_ev, bh, name="Equal-Vol (risk parity)")
    print(f"  CR={m_ev['cumulative_return']:.2f}%  SR={m_ev['sharpe_ratio']:.4f}")

    print("[4/N] Buy & Hold (equal weight) …")
    results["Buy & Hold (EW)"] = bh.tolist()
    m_bh = compute_full_metrics(bh, bh, name="Buy & Hold (EW)")
    print(f"  CR={m_bh['cumulative_return']:.2f}%  SR={m_bh['sharpe_ratio']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    all_names = list(results.keys())
    all_pvs   = list(results.values())
    rows = []
    for name, pvs in results.items():
        m = compute_full_metrics(pvs, bh, name=name)
        rows.append({
            "Strategy": name,
            "CR (%)":   m["cumulative_return"],
            "AR (%)":   m["annual_return"],
            "Sharpe":   m["sharpe_ratio"],
            "Sortino":  m["sortino_ratio"],
            "MDD (%)":  m["max_drawdown_pct"],
            "Rachev":   m["rachev_ratio"],
            "Calmar":   m["calmar_ratio"],
            "CVaR-5%":  m["cvar_5pct"],
            "Outperf%": m.get("outperf_overall", "—"),
        })

    df_res = pd.DataFrame(rows)
    print(f"\n{'═'*90}")
    print(df_res.to_string(index=False))
    df_res.to_csv("backtest_results/ablation_results.csv", index=False)
    print(f"\nSaved → backtest_results/ablation_results.csv")

    # ── Ablation impact chart ─────────────────────────────────────────────────
    _plot_ablation(results, unique_dates)
    _plot_baselines(results, unique_dates)


def _plot_ablation(results, dates):
    dates_dt = pd.to_datetime(dates)
    ablations = [k for k in results if "CPPO" in k]
    palette   = ["#2563EB","#F59E0B","#10B981","#EF4444","#8B5CF6","#94A3B8"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(ablations):
        pvs  = np.array(results[name])
        norm = pvs / pvs[0]
        lw   = 2.2 if name == "CPPO (all signals)" else 1.3
        ls   = "-" if name == "CPPO (all signals)" else "--"
        ax.plot(dates_dt[:len(pvs)], norm, label=name, color=palette[i % len(palette)],
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.set_title("Signal Ablation Study — CPPO Variants (2019–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value (start=1)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = "backtest_results/ablation_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")


def _plot_baselines(results, dates):
    dates_dt = pd.to_datetime(dates)
    strategies = {
        "CPPO (all signals)":    "#2563EB",
        "CPPO (neutral)":        "#94A3B8",
        "Momentum (top-10)":     "#F59E0B",
        "Equal-Vol (risk parity)":"#10B981",
        "Buy & Hold (EW)":       "#EF4444",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, color in strategies.items():
        if name not in results:
            continue
        pvs  = np.array(results[name])
        norm = pvs / pvs[0]
        lw   = 2.2 if "CPPO (all signals)" in name else 1.5
        ax.plot(dates_dt[:len(pvs)], norm, label=name, color=color, linewidth=lw, alpha=0.9)

    ax.set_title("Strategy Comparison — CPPO vs Baselines (2019–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value (start=1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = "backtest_results/baseline_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
