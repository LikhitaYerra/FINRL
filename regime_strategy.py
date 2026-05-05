"""
Regime-Switching Trading Strategy

Uses a Hidden Markov Model (HMM) with 3 states (bull / neutral / bear)
trained on market features to dynamically adjust:
  - Position sizing (aggressive / neutral / defensive)
  - Risk exposure (max holdings fraction per regime)
  - LLM signal weighting (amplify in trending, dampen in noisy regimes)

Architecture:
  1. Fit HMM on training data (returns + volatility + turbulence)
  2. Detect regime at each trading step
  3. Adjust env action via a regime-conditioned scaling layer
  4. Run CPPO agent with regime-scaled actions

This is a meta-strategy layer ABOVE the CPPO agent.
The CPPO still decides direction; the regime layer decides magnitude.

Usage:
    python regime_strategy.py
    python regime_strategy.py --plot
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
from hmmlearn import hmm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv
from finrl.config import INDICATORS


# ─── Constants ────────────────────────────────────────────────────────────────

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

# Regime action scaling: bull → aggressive, bear → defensive
REGIME_SCALE = {
    0: 1.3,   # bull  → amplify positions
    1: 1.0,   # neutral
    2: 0.5,   # bear  → halve positions
}
REGIME_LABEL = {0: "Bull", 1: "Neutral", 2: "Bear"}
REGIME_COLOR = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}




# ─── HMM Regime Detector ──────────────────────────────────────────────────────

class RegimeDetector:
    """
    3-state Gaussian HMM trained on:
      - Daily market return (mean of all stocks)
      - Rolling 10-day return volatility
      - Turbulence index (if available)

    States are relabeled post-hoc by mean return (bull=highest, bear=lowest).
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model    = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._state_map: dict = {}   # HMM state → (0=bull, 1=neutral, 2=bear)

    def fit(self, features: np.ndarray):
        """features: (T, D) array of market features per day."""
        self.model.fit(features)
        # Relabel states by mean return (feature 0)
        means = self.model.means_[:, 0]
        order = np.argsort(-means)  # descending: bull first
        self._state_map = {int(orig): int(new) for new, orig in enumerate(order)}
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict relabeled regime for each time step."""
        raw = self.model.predict(features)
        return np.array([self._state_map.get(s, 1) for s in raw])

    def predict_online(self, features: np.ndarray) -> int:
        """Predict current regime from a window ending at today."""
        raw = self.model.predict(features)
        return self._state_map.get(int(raw[-1]), 1)


def build_market_features(df: pd.DataFrame, dates: list) -> np.ndarray:
    """
    Build daily feature matrix for HMM:
      - mean daily return across all stocks
      - std of daily return (cross-sectional dispersion)
      - rolling 10-day volatility
      - turbulence (if present)
    """
    pivot  = df.pivot(index="date", columns="tic", values="close").sort_index()
    # Align to dates
    pivot  = pivot.reindex(dates).ffill()

    ret    = pivot.pct_change()
    mret   = ret.mean(axis=1).values
    xstd   = ret.std(axis=1).values
    vol10  = pd.Series(mret).rolling(10, min_periods=1).std().values

    features = np.column_stack([mret, xstd, vol10])
    # Replace NaN/inf with 0 and clip extreme values
    features = np.nan_to_num(features, nan=0.0, posinf=0.3, neginf=-0.3)
    features = np.clip(features, -0.3, 0.3)
    return features


# ─── Regime-Conditioned Backtest ──────────────────────────────────────────────

def run_regime_strategy(
    ac: _AC,
    trade_df: pd.DataFrame,
    unique_dates: list,
    regime_detector: RegimeDetector,
    features: np.ndarray,
) -> tuple[list, list]:
    """
    Run CPPO agent with regime-scaled actions.
    Returns (portfolio_values, regime_sequence).
    """
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs   = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    pvs   = [INITIAL_AMOUNT]
    regimes = []
    done  = False
    step  = 0

    while not done:
        # Detect current regime from features up to this step
        window = features[:min(step + 1, len(features))]
        regime = regime_detector.predict_online(window)
        scale  = REGIME_SCALE[regime]
        regimes.append(regime)

        # Scale action
        action = ac.act(torch.FloatTensor(obs)) * scale

        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        step += 1
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])

    return pvs[:len(unique_dates)], regimes[:len(unique_dates)-1]


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_regime_overlay(pvs, regimes, dates, bh, path):
    """Plot portfolio with regime background shading."""
    dates_dt = pd.to_datetime(dates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: equity curves
    norm_agent = np.array(pvs) / pvs[0]
    norm_bh    = np.array(bh)  / bh[0]
    n = min(len(norm_agent), len(dates_dt))

    ax1.plot(dates_dt[:n], norm_agent[:n], color="#2563EB", lw=2.2, label="CPPO + Regime Switch")
    ax1.plot(dates_dt[:n], norm_bh[:n],    color="#94A3B8", lw=1.5, ls="--", label="Buy & Hold (EW)")

    # Shade regime backgrounds
    regime_arr = np.array(regimes)
    for r_label, r_color in REGIME_COLOR.items():
        mask = (regime_arr == r_label)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        # Group consecutive indices
        groups = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        for g in groups:
            if len(g) == 0:
                continue
            ax1.axvspan(dates_dt[g[0]], dates_dt[min(g[-1]+1, n-1)],
                        alpha=0.08, color=r_color, label=f"_{REGIME_LABEL[r_label]}")

    # Dummy patches for legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=REGIME_COLOR[0], alpha=0.3, label="Bull regime"),
        Patch(facecolor=REGIME_COLOR[1], alpha=0.3, label="Neutral regime"),
        Patch(facecolor=REGIME_COLOR[2], alpha=0.3, label="Bear regime"),
    ]
    handles, labels = ax1.get_legend_handles_labels()
    handles += legend_patches
    labels  += [p.get_label() for p in legend_patches]
    ax1.legend(handles, labels, fontsize=8, loc="upper left")
    ax1.set_ylabel("Normalised Portfolio Value", fontsize=10)
    ax1.set_title("CPPO + Regime Switching vs Buy & Hold (2019–2023)", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Bottom: regime bar
    regime_colors_arr = [REGIME_COLOR[r] for r in regime_arr[:n]]
    for i in range(min(len(regime_arr), n-1)):
        ax2.axvspan(dates_dt[i], dates_dt[i+1], color=REGIME_COLOR[regime_arr[i]], alpha=0.7)
    ax2.set_ylabel("Regime", fontsize=9)
    ax2.set_yticks([])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    os.makedirs("backtest_results", exist_ok=True)

    print("Loading data …")
    train = pd.read_csv("train_data_multi_signal_2013_2018.csv")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")

    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)

    unique_dates_train = sorted(train["date"].unique())
    unique_dates_trade = sorted(trade["date"].unique())
    n = len(unique_dates_trade)

    date_to_idx = {d: i for i, d in enumerate(unique_dates_trade)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")

    # Build features for HMM fitting (training period)
    print("Building market features for HMM …")
    train_feats = build_market_features(train, unique_dates_train)
    trade_feats = build_market_features(
        trade.reset_index(), unique_dates_trade
    )

    # Fit HMM on training data only (no look-ahead)
    print("Fitting HMM on training data …")
    detector = RegimeDetector(n_states=3)
    detector.fit(train_feats)

    # Predict regimes for trade period
    trade_regimes = detector.predict(trade_feats)
    regime_counts = {REGIME_LABEL[r]: int((trade_regimes == r).sum()) for r in range(3)}
    print(f"  Trade period regime distribution: {regime_counts}")

    # Load model
    print(f"Loading model: {args.model}")
    ac = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    # Buy and hold baseline
    bh = INITIAL_AMOUNT * (
        trade.reset_index().groupby("date")["close"].mean().sort_index().values /
        trade.reset_index().groupby("date")["close"].mean().sort_index().values[0]
    )[:n]

    # Plain CPPO (no regime switching)
    print("\nRunning plain CPPO …")
    from env_stocktrading_multi_signal import StockTradingEnv
    env = StockTradingEnv(df=trade, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    pvs_plain = [INITIAL_AMOUNT]
    done = False
    while not done:
        action = ac.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs_plain.append(float(env.asset_memory[-1]) if env.asset_memory else pvs_plain[-1])
    pvs_plain = pvs_plain[:n]

    # Regime-switching CPPO
    print("Running regime-switching CPPO …")
    pvs_regime, regimes = run_regime_strategy(
        ac, trade, unique_dates_trade, detector, trade_feats
    )

    # Metrics
    m_plain  = compute_full_metrics(pvs_plain,  bh, name="CPPO (plain)")
    m_regime = compute_full_metrics(pvs_regime, bh, name="CPPO + Regime Switch")
    m_bh     = compute_full_metrics(bh,         bh, name="Buy & Hold (EW)")

    print_metrics(m_plain)
    print_metrics(m_regime)

    print(f"\n{'─'*60}")
    print(f"  Regime Switching Impact")
    print(f"{'─'*60}")
    print(f"  ΔCumulative Return : {m_regime['cumulative_return']-m_plain['cumulative_return']:+.2f} pp")
    print(f"  ΔSharpe Ratio      : {m_regime['sharpe_ratio']-m_plain['sharpe_ratio']:+.4f}")
    print(f"  ΔMDD               : {m_regime['max_drawdown_pct']-m_plain['max_drawdown_pct']:+.2f} pp")
    print(f"  ΔRachev            : {m_regime['rachev_ratio']-m_plain['rachev_ratio']:+.4f}")

    # Save
    pd.DataFrame({
        "date":        unique_dates_trade[:n],
        "cppo_plain":  pvs_plain,
        "cppo_regime": pvs_regime,
        "buy_hold":    bh[:n],
        "regime":      [REGIME_LABEL[r] for r in trade_regimes[:n-1]] + ["Neutral"],
    }).to_csv("backtest_results/regime_results.csv", index=False)
    print("\nSaved → backtest_results/regime_results.csv")

    if args.plot:
        plot_regime_overlay(pvs_regime, regimes, unique_dates_trade, bh,
                            "backtest_results/regime_overlay.png")


if __name__ == "__main__":
    main()
