"""
Attention Weight Interpretability

Loads a trained CPPO+SAM model and visualises what the cross-attention
layer learns over the trading period:
  - How attention weights evolve over time (which signal is queried when)
  - Correlation between attention weight and subsequent performance
  - Which signals receive high attention in bull vs bear regimes

Since we may not have a trained SAM model yet, this script also supports
running on the baseline MLP model with a "pseudo-attention" approximation
via gradient-based attribution (integrated gradients over signal dimensions).

Usage:
    python attention_viz.py --mode gradient   # works with any model
    python attention_viz.py --mode sam        # requires SAM model
"""

from __future__ import annotations

import argparse
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
import seaborn as sns

from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv
from finrl.config import INDICATORS


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

SIGNAL_NAMES  = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
SIGNAL_LABELS = ["Sentiment", "Risk", "Confidence", "Vol Forecast"]

FIG_DIR  = "paper/figures"
DATA_DIR = "backtest_results"

# Indices of signal block in state vector
N, K, S = STOCK_DIM, len(INDICATORS), 4
SIGNAL_START = 1 + 2*N + K*N   # = 1 + 60 + 240 = 301
SIGNAL_END   = SIGNAL_START + S*N  # = 421


# ─── Gradient-based signal attribution ───────────────────────────────────────

def gradient_attribution(
    model: nn.Module,
    obs_tensor: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Integrated gradients of the policy output w.r.t. signal dimensions.

    For each observation, compute: ∂mean_action / ∂signal_i,
    integrated from baseline (neutral signals = 3.0) to actual signal.

    Returns: (T, 4) array of per-signal attributions.
    """
    T = obs_tensor.shape[0]
    attrs = np.zeros((T, S), dtype=np.float32)

    # Baseline: neutral signals
    baseline = obs_tensor.clone()
    baseline[:, SIGNAL_START:SIGNAL_END] = 3.0

    for t in range(T):
        obs  = obs_tensor[t:t+1]
        base = baseline[t:t+1]

        # Integrated gradients: sum over N_STEPS interpolations
        grad_sum = torch.zeros(1, S*N)
        for alpha in np.linspace(0, 1, n_steps):
            interp = (base + alpha * (obs - base)).requires_grad_(True)
            action = model.pi.mu_net(interp)
            loss   = action.mean()
            loss.backward()
            grad_sum += interp.grad[:, SIGNAL_START:SIGNAL_END].detach()

        # Average per signal dimension (mean across N stocks)
        grad_per_signal = grad_sum.view(1, S, N)
        delta_signal    = (obs - base)[:, SIGNAL_START:SIGNAL_END].view(1, S, N)
        ig = (grad_per_signal / n_steps) * delta_signal
        attrs[t] = ig.abs().mean(dim=2).squeeze(0).numpy()

    return attrs


def fast_gradient_sensitivity(
    model: nn.Module,
    obs_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Faster approximation: plain gradient magnitude of policy output
    w.r.t. each signal dimension (sum |∂action/∂signal_i|).
    
    Returns: (T, 4) per-signal sensitivity.
    """
    T     = obs_tensor.shape[0]
    attrs = np.zeros((T, S), dtype=np.float32)

    for t in range(0, T, 10):  # batch by 10 for speed
        batch = obs_tensor[t:t+10].clone().requires_grad_(True)
        action = model.pi.mu_net(batch)
        loss   = action.sum()
        loss.backward()
        grad = batch.grad[:, SIGNAL_START:SIGNAL_END]   # (batch, S*N)
        grad_per_sig = grad.view(-1, S, N).abs().mean(dim=2)  # (batch, S)
        n_batch = min(10, T - t)
        attrs[t:t+n_batch] = grad_per_sig[:n_batch].detach().numpy()

    return attrs


# ─── Run episode and collect observations ────────────────────────────────────

def collect_observations(model, trade_df, unique_dates):
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    obs_list = [obs.copy()]
    pvs  = [INITIAL_AMOUNT]
    done = False

    while not done:
        action = model.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        obs_list.append(obs.copy())
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])

    obs_arr = np.array(obs_list[:len(unique_dates)])
    pvs_arr = np.array(pvs[:len(unique_dates)])
    return obs_arr, pvs_arr


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_attribution_over_time(attrs, dates, pvs, out):
    dates_dt = pd.to_datetime(dates)
    T = min(len(attrs), len(dates_dt))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1.5]})

    # Stacked area of attribution weights
    palette = ["#2563EB", "#EF4444", "#10B981", "#F59E0B"]
    norm    = attrs[:T] / (attrs[:T].sum(axis=1, keepdims=True) + 1e-9)

    prev = np.zeros(T)
    for i, (label, color) in enumerate(zip(SIGNAL_LABELS, palette)):
        ax1.fill_between(dates_dt[:T], prev, prev + norm[:, i],
                         alpha=0.7, color=color, label=label)
        prev = prev + norm[:, i]

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Normalised Signal Attribution", fontsize=10)
    ax1.set_title("Policy Signal Attribution over Time — CPPO Agent",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(alpha=0.2)

    # Portfolio value
    norm_pv = pvs[:T] / pvs[0]
    ax2.plot(dates_dt[:T], norm_pv, color="#1E293B", lw=1.5)
    ax2.fill_between(dates_dt[:T], 1, norm_pv,
                     where=norm_pv >= 1, alpha=0.2, color="#10B981")
    ax2.fill_between(dates_dt[:T], 1, norm_pv,
                     where=norm_pv < 1,  alpha=0.2, color="#EF4444")
    ax2.set_ylabel("Normalised Portfolio Value")
    ax2.axhline(1, color="black", lw=0.8, ls="--")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 11 → {out}")


def plot_attribution_by_regime(attrs, dates, pvs, out):
    """Heatmap: mean attribution per signal in each year/quarter."""
    dates_dt = pd.to_datetime(dates)
    T = min(len(attrs), len(dates_dt))

    norm  = attrs[:T] / (attrs[:T].sum(axis=1, keepdims=True) + 1e-9)
    df    = pd.DataFrame(norm, columns=SIGNAL_LABELS, index=dates_dt[:T])
    df["year"] = df.index.year
    yearly = df.groupby("year")[SIGNAL_LABELS].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(yearly.T, ax=ax, annot=True, fmt=".2f",
                cmap="Blues", cbar_kws={"shrink": 0.8})
    ax.set_title("Mean Signal Attribution Weight by Year",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("LLM Signal")
    ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 12 → {out}")


def plot_attribution_correlation(attrs, pvs, out):
    """Does high attribution on a signal predict better/worse returns?"""
    T = min(len(attrs), len(pvs) - 1)
    fwd_ret = np.diff(pvs[:T+1]) / pvs[:T]
    norm    = attrs[:T] / (attrs[:T].sum(axis=1, keepdims=True) + 1e-9)

    from scipy import stats
    ics = []
    for i, label in enumerate(SIGNAL_LABELS):
        ic, pval = stats.spearmanr(norm[:, i], fwd_ret)
        ics.append({"Signal": label, "IC": ic, "p-value": pval})
    df_ic = pd.DataFrame(ics)
    print("\n  Attribution-Return IC:")
    print(df_ic.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#10B981" if ic >= 0 else "#EF4444" for ic in df_ic["IC"]]
    bars = ax.bar(df_ic["Signal"], df_ic["IC"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for bar, row in zip(bars, ics):
        sig = "**" if row["p-value"] < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                row["IC"] + 0.001 * np.sign(row["IC"]),
                f"{row['IC']:.3f}{sig}", ha="center", fontsize=9)
    ax.set_ylabel("Spearman IC (attribution weight vs 1-day return)")
    ax.set_title("Does High Attribution Predict Better Returns?",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 13 → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--mode",  choices=["gradient"], default="gradient")
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

    print(f"Loading model: {args.model}")
    model = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    print("Collecting observations …")
    obs_arr, pvs_arr = collect_observations(model, trade, unique_dates)

    print("Computing gradient-based signal attribution (fast approx) …")
    obs_tensor = torch.FloatTensor(obs_arr)
    attrs = fast_gradient_sensitivity(model, obs_tensor)

    print("Generating figures …")
    plot_attribution_over_time(attrs, unique_dates, pvs_arr,
                               f"{FIG_DIR}/fig11_attribution_time.png")
    plot_attribution_by_regime(attrs, unique_dates, pvs_arr,
                               f"{FIG_DIR}/fig12_attribution_by_year.png")
    plot_attribution_correlation(attrs, pvs_arr,
                                 f"{FIG_DIR}/fig13_attribution_ic.png")

    # Save attribution data
    pd.DataFrame(attrs, columns=SIGNAL_LABELS,
                 index=pd.to_datetime(unique_dates[:len(attrs)])).to_csv(
        f"{DATA_DIR}/signal_attribution.csv"
    )
    print(f"  Saved → {DATA_DIR}/signal_attribution.csv")


if __name__ == "__main__":
    main()
