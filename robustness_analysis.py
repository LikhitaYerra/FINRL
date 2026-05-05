"""
Robustness Analysis

Two experiments that a top-tier paper requires:

A) Sub-period breakdown
   Splits 2019-2023 into economically meaningful regimes:
     - Pre-COVID     (2019-01 to 2020-02)
     - COVID crash   (2020-03 to 2020-04)
     - Recovery bull (2020-05 to 2021-12)
     - Rate hike bear(2022-01 to 2022-12)
     - 2023 rally    (2023-01 to 2023-12)

B) Transaction cost sensitivity
   Sweeps buy/sell cost from 0.5× to 20× baseline (0.1%)
   Shows at what cost level the agent breaks even with B&H.

Usage:
    python robustness_analysis.py
"""

from __future__ import annotations

import json
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

from metrics_extended import compute_full_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv
try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]


# ─── Config ───────────────────────────────────────────────────────────────────

INITIAL_AMOUNT = 1_000_000
STOCK_DIM      = 30
STATE_SPACE    = 1 + 2 * STOCK_DIM + (len(INDICATORS) + 4) * STOCK_DIM
ACTION_SPACE   = STOCK_DIM
BASE_COST      = 0.001  # 0.1% per trade

SUB_PERIODS = {
    "Pre-COVID\n(2019–Feb 2020)":       ("2019-01-01", "2020-02-28"),
    "COVID Crash\n(Mar–Apr 2020)":       ("2020-03-01", "2020-04-30"),
    "Recovery Bull\n(May 2020–Dec 2021)":("2020-05-01", "2021-12-31"),
    "Rate Hike Bear\n(2022)":            ("2022-01-01", "2022-12-31"),
    "2023 Rally\n(2023)":                ("2023-01-01", "2023-12-31"),
}

COST_MULTIPLIERS = [0.5, 1, 2, 5, 10, 15, 20]

FIG_DIR  = "paper/figures"
DATA_DIR = "backtest_results"


# ─── Run agent ────────────────────────────────────────────────────────────────

def run_agent_with_cost(ac, trade_df, unique_dates, cost_mult=1.0):
    cost = BASE_COST * cost_mult
    env_kwargs = dict(
        hmax=100, initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0]*STOCK_DIM,
        buy_cost_pct=[cost]*STOCK_DIM, sell_cost_pct=[cost]*STOCK_DIM,
        reward_scaling=1e-4, state_space=STATE_SPACE, action_space=ACTION_SPACE,
        tech_indicator_list=INDICATORS, turbulence_threshold=380,
        drawdown_penalty=0.1, make_plots=False, print_verbosity=999,
    )
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **env_kwargs)

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


def bh_with_cost(trade_df, unique_dates, cost_mult=1.0):
    """Simple daily equal-weight rebalance with transaction cost."""
    pivot = trade_df.pivot(index="date", columns="tic", values="close").sort_index()
    pivot = pivot.reindex(unique_dates).ffill()
    ret   = pivot.pct_change()
    # B&H doesn't rebalance daily — just buy-and-hold, no ongoing costs
    # Compute raw equal-weight B&H
    avg = pivot.mean(axis=1)
    return INITIAL_AMOUNT * (avg / avg.iloc[0]).values


# ─── Sub-period analysis ──────────────────────────────────────────────────────

def sub_period_analysis(pvs_full, bh_full, unique_dates):
    """
    Computes per-period Sharpe, return, drawdown for agent vs B&H.
    """
    date_index = pd.DatetimeIndex(unique_dates)
    pvs_arr    = np.array(pvs_full)
    bh_arr     = np.array(bh_full)

    rows = []
    for period_name, (start, end) in SUB_PERIODS.items():
        mask = (date_index >= start) & (date_index <= end)
        if mask.sum() < 5:
            continue

        idx       = np.where(mask)[0]
        pv_slice  = pvs_arr[idx]
        bh_slice  = bh_arr[idx]

        # Normalise each period to start at 1M
        pv_norm = pv_slice * (INITIAL_AMOUNT / pv_slice[0])
        bh_norm = bh_slice * (INITIAL_AMOUNT / bh_slice[0])

        m_agent = compute_full_metrics(pv_norm, bh_norm, name=period_name)
        m_bh    = compute_full_metrics(bh_norm, bh_norm, name=f"{period_name} B&H")

        rows.append({
            "Period":            period_name.replace("\n", " "),
            "Days":              len(idx),
            "Agent CR (%)":      m_agent["cumulative_return"],
            "B&H CR (%)":        m_bh["cumulative_return"],
            "Δ CR (pp)":         m_agent["cumulative_return"] - m_bh["cumulative_return"],
            "Agent Sharpe":      m_agent["sharpe_ratio"],
            "B&H Sharpe":        m_bh["sharpe_ratio"],
            "Agent MDD (%)":     m_agent["max_drawdown_pct"],
            "B&H MDD (%)":       m_bh["max_drawdown_pct"],
        })

    return pd.DataFrame(rows)


# ─── Transaction cost sweep ───────────────────────────────────────────────────

def txcost_sweep(ac, trade_df, unique_dates):
    results = []
    for mult in COST_MULTIPLIERS:
        cost_pct = BASE_COST * mult * 100
        print(f"  Cost {cost_pct:.3f}% per trade (×{mult}) …")
        pvs_agent = run_agent_with_cost(ac, trade_df, unique_dates, cost_mult=mult)
        pvs_bh    = bh_with_cost(trade_df, unique_dates, cost_mult=mult)[:len(unique_dates)]
        m_agent   = compute_full_metrics(pvs_agent, pvs_bh, name=f"CPPO×{mult}")
        m_bh      = compute_full_metrics(pvs_bh,    pvs_bh, name=f"B&H×{mult}")
        results.append({
            "cost_mult":    mult,
            "cost_pct":     cost_pct,
            "agent_cr":     m_agent["cumulative_return"],
            "bh_cr":        m_bh["cumulative_return"],
            "delta_cr":     m_agent["cumulative_return"] - m_bh["cumulative_return"],
            "agent_sharpe": m_agent["sharpe_ratio"],
            "bh_sharpe":    m_bh["sharpe_ratio"],
            "agent_mdd":    m_agent["max_drawdown_pct"],
            "agent_rachev": m_agent["rachev_ratio"],
        })
    return pd.DataFrame(results)


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_sub_period(df_sp, pvs_full, bh_full, unique_dates, out_bar, out_equity):
    dates_dt = pd.to_datetime(unique_dates)
    n = min(len(pvs_full), len(dates_dt))

    # Equity curve with shaded sub-periods
    fig, ax = plt.subplots(figsize=(13, 5))
    palette = ["#E0F2FE", "#FEF3C7", "#D1FAE5", "#FEE2E2", "#EDE9FE"]
    for i, (period_name, (start, end)) in enumerate(SUB_PERIODS.items()):
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   alpha=0.25, color=palette[i % len(palette)],
                   label=period_name.replace("\n", " "))

    norm_agent = np.array(pvs_full[:n]) / pvs_full[0]
    norm_bh    = np.array(bh_full[:n])  / bh_full[0]
    ax.plot(dates_dt[:n], norm_agent, color="#2563EB", lw=2.2, label="CPPO (LLM signals)", zorder=5)
    ax.plot(dates_dt[:n], norm_bh,    color="#94A3B8", lw=1.5, ls="--", label="Buy & Hold", zorder=5)
    ax.set_title("CPPO vs Buy & Hold — Sub-Period Shading (2019–2023)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value (start=1)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_equity, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 8 → {out_equity}")

    # Bar chart: ΔCR per sub-period
    fig, ax = plt.subplots(figsize=(10, 4))
    periods = df_sp["Period"].tolist()
    delta   = df_sp["Δ CR (pp)"].tolist()
    colors  = ["#10B981" if d >= 0 else "#EF4444" for d in delta]
    bars    = ax.bar(periods, delta, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for bar, val in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3 * np.sign(val),
                f"{val:+.1f}", ha="center", va="bottom" if val > 0 else "top", fontsize=9)
    ax.set_ylabel("CPPO − B&H Cumulative Return (pp)", fontsize=10)
    ax.set_title("Alpha vs Buy & Hold by Sub-Period", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 9 → {out_bar}")


def plot_txcost(df_tc, out: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("agent_cr", "bh_cr",     "Cumulative Return (%)",       "CR"),
        ("agent_sharpe", "bh_sharpe", "Sharpe Ratio",             "Sharpe"),
        ("delta_cr", None,         "CPPO − B&H (pp)",            "ΔCR vs B&H"),
    ]
    for ax, (agent_col, bh_col, ylabel, title) in zip(axes, metrics):
        ax.plot(df_tc["cost_pct"], df_tc[agent_col], color="#2563EB",
                marker="o", lw=2, label="CPPO")
        if bh_col:
            ax.plot(df_tc["cost_pct"], df_tc[bh_col], color="#94A3B8",
                    marker="s", lw=1.5, ls="--", label="Buy & Hold")
        if agent_col == "delta_cr":
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.fill_between(df_tc["cost_pct"], df_tc["delta_cr"], 0,
                            where=df_tc["delta_cr"] >= 0, alpha=0.15, color="#10B981")
            ax.fill_between(df_tc["cost_pct"], df_tc["delta_cr"], 0,
                            where=df_tc["delta_cr"] < 0, alpha=0.15, color="#EF4444")
        ax.set_xlabel("Transaction Cost per Trade (%)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        if bh_col:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Mark break-even cost
    breakeven = df_tc[df_tc["delta_cr"] >= 0]["cost_pct"].max()
    if not pd.isna(breakeven):
        axes[2].axvline(breakeven, color="#F59E0B", lw=1.5, ls=":",
                        label=f"Break-even: {breakeven:.3f}%")
        axes[2].legend(fontsize=8)

    fig.suptitle("Transaction Cost Sensitivity Analysis (CPPO vs Buy & Hold)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 10 → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    args = parser.parse_args()

    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    n = len(unique_dates)

    print(f"Loading model: {args.model}")
    ac = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    # ── Sub-period analysis ────────────────────────────────────────────────────
    print("\n[A] Sub-period analysis …")
    pvs_full = run_agent_with_cost(ac, trade, unique_dates, cost_mult=1.0)
    bh_full  = bh_with_cost(trade, unique_dates, cost_mult=1.0)[:n]
    df_sp    = sub_period_analysis(pvs_full, bh_full, unique_dates)
    print(df_sp.to_string(index=False))
    df_sp.to_csv(f"{DATA_DIR}/sub_period_results.csv", index=False)
    print(f"  Saved → {DATA_DIR}/sub_period_results.csv")

    # ── Transaction cost sweep ─────────────────────────────────────────────────
    print("\n[B] Transaction cost sensitivity sweep …")
    df_tc = txcost_sweep(ac, trade, unique_dates)
    print(df_tc[["cost_pct","agent_cr","bh_cr","delta_cr","agent_sharpe"]].to_string(index=False))
    df_tc.to_csv(f"{DATA_DIR}/txcost_sensitivity.csv", index=False)
    print(f"  Saved → {DATA_DIR}/txcost_sensitivity.csv")

    # Break-even cost
    be_row = df_tc[df_tc["delta_cr"] >= 0]
    if not be_row.empty:
        be_cost = be_row["cost_pct"].max()
        print(f"\n  Agent beats B&H up to {be_cost:.3f}% per-trade cost ({be_cost/BASE_COST/100:.0f}× baseline)")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n[C] Generating figures …")
    plot_sub_period(df_sp, pvs_full, bh_full, unique_dates,
                    f"{FIG_DIR}/fig9_alpha_by_period.png",
                    f"{FIG_DIR}/fig8_sub_period_equity.png")
    plot_txcost(df_tc, f"{FIG_DIR}/fig10_txcost_sensitivity.png")


if __name__ == "__main__":
    main()
