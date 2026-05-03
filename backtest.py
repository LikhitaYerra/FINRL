"""
Backtest the trained CPPO agent on the 2019-2023 trade data.

Outputs:
  - backtest_results/portfolio_value.csv   daily portfolio values
  - backtest_results/metrics.json          summary performance metrics
  - backtest_results/portfolio_plot.png    equity curve vs buy-and-hold
  - backtest_results/drawdown_plot.png     drawdown over time

Usage:
    python backtest.py
    python backtest.py --model trained_models/agent_cppo_multi_signal_30_epochs.pth
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

from env_stocktrading_multi_signal import StockTradingEnv
from finrl.config import INDICATORS

# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic network (must match training architecture)
# ─────────────────────────────────────────────────────────────────────────────

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net  = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        mu  = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            mu, _ = self.pi(obs)
        return mu.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration  (must match training)
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_AMOUNT    = 1_000_000
HMAX              = 100
STOCK_DIM         = 30
STATE_SPACE       = 1 + 2 * STOCK_DIM + (len(INDICATORS) + 4) * STOCK_DIM  # 421
ACTION_SPACE      = STOCK_DIM

BUY_COST_PCT  = [0.001] * STOCK_DIM
SELL_COST_PCT = [0.001] * STOCK_DIM
NUM_SHARES    = [0]     * STOCK_DIM

ENV_KWARGS = dict(
    hmax                = HMAX,
    initial_amount      = INITIAL_AMOUNT,
    num_stock_shares    = NUM_SHARES,
    buy_cost_pct        = BUY_COST_PCT,
    sell_cost_pct       = SELL_COST_PCT,
    reward_scaling      = 1e-4,
    state_space         = STATE_SPACE,
    action_space        = ACTION_SPACE,
    tech_indicator_list = INDICATORS,
    turbulence_threshold= 380,
    drawdown_penalty    = 0.1,
    make_plots          = False,
    print_verbosity     = 999,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(portfolio_values: list, dates: list, initial: float) -> dict:
    pv  = np.array(portfolio_values, dtype=float)
    ret = np.diff(pv) / pv[:-1]

    cumulative_return = (pv[-1] / initial - 1) * 100
    annual_return     = ((pv[-1] / initial) ** (252 / len(pv)) - 1) * 100

    # Sharpe (annualised, rf=0)
    sharpe = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252)

    # Max drawdown
    running_max   = np.maximum.accumulate(pv)
    drawdowns     = (pv - running_max) / running_max
    max_drawdown  = drawdowns.min() * 100

    # Calmar
    calmar = annual_return / abs(max_drawdown + 1e-9)

    # Sortino
    neg_ret   = ret[ret < 0]
    down_std  = neg_ret.std() + 1e-9
    sortino   = (ret.mean() / down_std) * np.sqrt(252)

    return dict(
        initial_capital   = initial,
        final_value       = float(pv[-1]),
        cumulative_return = round(cumulative_return, 4),
        annual_return     = round(annual_return, 4),
        sharpe_ratio      = round(sharpe, 4),
        sortino_ratio     = round(sortino, 4),
        max_drawdown_pct  = round(max_drawdown, 4),
        calmar_ratio      = round(calmar, 4),
        n_trading_days    = len(pv),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Buy-and-hold baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_buy_hold(trade_df: pd.DataFrame, initial: float) -> np.ndarray:
    """Equal-weight buy-and-hold across all tickers."""
    first_day = trade_df[trade_df["date"] == trade_df["date"].min()]
    last_day  = trade_df[trade_df["date"] == trade_df["date"].max()]

    prices_per_date = (
        trade_df.groupby("date")["close"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    # Normalise to initial capital
    base = prices_per_date["close"].iloc[0]
    bh   = initial * prices_per_date["close"].values / base
    return bh


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

STYLE = dict(linewidth=1.6, alpha=0.9)

def plot_equity(dates, agent_pv, bh_pv, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, np.array(agent_pv) / 1e6, label="CPPO Agent", color="#2563EB", **STYLE)
    ax.plot(dates[:len(bh_pv)], np.array(bh_pv) / 1e6, label="Buy & Hold", color="#F59E0B",
            linestyle="--", **STYLE)
    ax.set_title("Portfolio Value — CPPO Agent vs Buy & Hold", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_drawdown(dates, agent_pv, out_path):
    pv  = np.array(agent_pv)
    rm  = np.maximum.accumulate(pv)
    dd  = (pv - rm) / rm * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, dd, 0, color="#EF4444", alpha=0.4, label="Drawdown")
    ax.plot(dates, dd, color="#EF4444", linewidth=1)
    ax.set_title("Portfolio Drawdown — CPPO Agent", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(model_path: str):
    os.makedirs("backtest_results", exist_ok=True)

    # ── Load trade data ──────────────────────────────────────────────────────
    print("Loading trade data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")

    # Re-index by day index (matching training)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")

    for col, default in [
        ("llm_sentiment", 3), ("llm_risk", 3),
        ("llm_confidence", 3), ("llm_volatility_forecast", 3),
    ]:
        trade[col] = trade.get(col, default).fillna(default)

    stock_dim = trade["tic"].nunique()
    print(f"  Stocks: {stock_dim} | Days: {len(unique_dates)}")

    # ── Build environment ────────────────────────────────────────────────────
    env = StockTradingEnv(df=trade, stock_dim=stock_dim, **ENV_KWARGS)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    ac = MLPActorCritic(obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)
    ac.load_state_dict(state_dict)
    ac.eval()
    print("  Model loaded.")

    # ── Run episode ──────────────────────────────────────────────────────────
    print("Running backtest …")
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs = np.array(obs, dtype=np.float32)

    portfolio_values = [INITIAL_AMOUNT]
    done = False

    while not done:
        action = ac.act(torch.FloatTensor(obs))
        step_out = env.step(action)
        obs, reward, terminated, truncated, info = step_out
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated

        pv = env.asset_memory[-1] if hasattr(env, "asset_memory") and env.asset_memory else \
             (info.get("portfolio_value", portfolio_values[-1]) if isinstance(info, dict) else portfolio_values[-1])
        portfolio_values.append(float(pv))

    # Trim to same length as dates
    portfolio_values = portfolio_values[:len(unique_dates)]
    dates_dt = pd.to_datetime(unique_dates)

    # ── Buy & Hold baseline ──────────────────────────────────────────────────
    bh_values = compute_buy_hold(trade, INITIAL_AMOUNT)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(portfolio_values, unique_dates, INITIAL_AMOUNT)
    print("\n── Agent Metrics ──────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    bh_metrics = compute_metrics(bh_values.tolist(), unique_dates, INITIAL_AMOUNT)
    print("\n── Buy & Hold Metrics ─────────────────────────────────────")
    for k, v in bh_metrics.items():
        print(f"  {k:<22}: {v}")

    # Save
    pd.DataFrame({
        "date": unique_dates,
        "cppo_value": portfolio_values,
        "bh_value": bh_values.tolist()[:len(unique_dates)],
    }).to_csv("backtest_results/portfolio_value.csv", index=False)

    with open("backtest_results/metrics.json", "w") as f:
        json.dump({"agent": metrics, "buy_hold": bh_metrics}, f, indent=2)
    print("\n  Saved: backtest_results/metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_equity(dates_dt, portfolio_values, bh_values, "backtest_results/portfolio_plot.png")
    plot_drawdown(dates_dt, portfolio_values, "backtest_results/drawdown_plot.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    args = parser.parse_args()
    run_backtest(args.model)
