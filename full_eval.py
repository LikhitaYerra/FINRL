"""
Full evaluation pipeline — runs all experiments and generates all paper figures.

Experiments:
  1. Main comparison  : CPPO (LLM) vs CPPO (neutral) vs Buy & Hold
  2. SAM ablation     : SAM-PPO (attention) vs CPPO (MLP)  ← architecture
  3. SAC comparison   : SAC (LLM) vs CPPO (LLM)            ← algorithm
  4. Multi-seed stats : mean ± std, Wilcoxon p-value
  5. OOS evaluation   : 2024-2025 neutral vs live signals
  6. Sub-period       : per-year Sharpe decomposition

Usage:
    python full_eval.py                 # run all
    python full_eval.py --skip_oos      # skip 2024-2025 download
    python full_eval.py --skip_sac      # if SAC hasn't trained
    python full_eval.py --skip_sam      # if SAM hasn't trained
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

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


# ─── Utilities ────────────────────────────────────────────────────────────────

def run_cppo(model, trade_df, n):
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    obs  = np.array(env.reset()[0], dtype=np.float32)
    pvs  = [INITIAL_AMOUNT]
    done = False
    while not done:
        a = model.act(torch.FloatTensor(obs))
        obs, _, term, trunc, _ = env.step(a)
        obs  = np.array(obs, dtype=np.float32)
        done = term or trunc
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])
    return pvs[:n]


def bh_series(trade_df, n):
    return (INITIAL_AMOUNT *
            trade_df.reset_index().groupby("date")["close"].mean().sort_index().values /
            trade_df.reset_index().groupby("date")["close"].mean().sort_index().values[0])[:n]


def load_trade(signal_mode="llm"):
    df = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    for col in SIGNAL_COLS:
        df[col] = df.get(col, 3.0).fillna(3.0)
    if signal_mode == "neutral":
        for col in SIGNAL_COLS:
            df[col] = 3.0
    dates = sorted(df["date"].unique())
    df["new_idx"] = df["date"].map({d: i for i, d in enumerate(dates)})
    return df.set_index("new_idx"), dates


def save_figure(fig, fname):
    os.makedirs("paper/figures", exist_ok=True)
    path = f"paper/figures/{fname}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Experiment 1: Main comparison ────────────────────────────────────────────

def exp1_main_comparison(trade_llm, trade_neutral, unique_dates):
    print("\n[Exp 1] Main comparison: LLM vs Neutral vs B&H …")
    n    = len(unique_dates)
    bh   = bh_series(trade_llm, n)
    model = load_cppo_model(
        "trained_models/agent_cppo_multi_signal_30_epochs.pth",
        obs_dim=STATE_SPACE, act_dim=ACTION_SPACE,
    )
    pvs_llm     = run_cppo(model, trade_llm,     n)
    pvs_neutral = run_cppo(model, trade_neutral,  n)

    m_llm     = compute_full_metrics(pvs_llm,     bh, name="CPPO+LLM")
    m_neutral = compute_full_metrics(pvs_neutral, bh, name="CPPO+Neutral")
    m_bh      = compute_full_metrics(bh,          bh, name="B&H")

    # Save metrics
    os.makedirs("backtest_results", exist_ok=True)
    pd.DataFrame([m_llm, m_neutral, m_bh]).to_csv(
        "backtest_results/main_comparison.csv", index=False)
    print(f"  CPPO+LLM Sharpe={m_llm['sharpe_ratio']:.4f}  "
          f"CPPO+Neutral Sharpe={m_neutral['sharpe_ratio']:.4f}")

    # Figure
    dates_dt = pd.to_datetime(unique_dates[:n])
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dates_dt, np.array(pvs_llm)/pvs_llm[0],
            "#2563EB", lw=2.2, label=f"CPPO + LLM Signals  (Sharpe {m_llm['sharpe_ratio']:.2f})")
    ax.plot(dates_dt, np.array(pvs_neutral)/pvs_neutral[0],
            "#F59E0B", lw=1.8, ls="--", label=f"CPPO + Neutral  (Sharpe {m_neutral['sharpe_ratio']:.2f})")
    ax.plot(dates_dt, bh[:n]/bh[0],
            "#94A3B8", lw=1.5, ls=":", label="Buy & Hold (EW)")
    ax.set_title("Main Result: LLM Signals vs Neutral Signals", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_figure(fig, "fig1_main_comparison.png")
    return m_llm, m_neutral, m_bh, pvs_llm, bh, unique_dates


# ─── Experiment 2: SAM ablation ───────────────────────────────────────────────

def exp2_sam_ablation(trade_llm, unique_dates, pvs_cppo, bh):
    sam_path = "trained_models/agent_sam_ppo_30e.pth"
    if not os.path.exists(sam_path):
        print(f"\n[Exp 2] SAM model not found, skipping ({sam_path})")
        return None
    print("\n[Exp 2] SAM (attention) vs CPPO (MLP) …")
    from backtest_sam import load_sam_model
    n   = len(unique_dates)
    sam = load_sam_model(sam_path)
    pvs_sam = run_cppo(sam, trade_llm, n)

    m_cppo = compute_full_metrics(pvs_cppo, bh, name="CPPO-MLP")
    m_sam  = compute_full_metrics(pvs_sam,  bh, name="SAM-PPO")
    pd.DataFrame([m_cppo, m_sam]).to_csv("backtest_results/sam_comparison.csv", index=False)
    print(f"  CPPO Sharpe={m_cppo['sharpe_ratio']:.4f}  SAM Sharpe={m_sam['sharpe_ratio']:.4f}  "
          f"Δ={m_sam['sharpe_ratio']-m_cppo['sharpe_ratio']:+.4f}")

    dates_dt = pd.to_datetime(unique_dates[:n])
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dates_dt, np.array(pvs_cppo)/pvs_cppo[0], "#2563EB", lw=2.0,
            label=f"CPPO (MLP)  Sharpe={m_cppo['sharpe_ratio']:.2f}")
    ax.plot(dates_dt, np.array(pvs_sam)/pvs_sam[0], "#10B981", lw=2.0, ls="--",
            label=f"SAM-PPO (attention)  Sharpe={m_sam['sharpe_ratio']:.2f}")
    ax.plot(dates_dt, bh[:n]/bh[0], "#94A3B8", lw=1.5, ls=":", label="Buy & Hold (EW)")
    ax.set_title("Architecture Ablation: SAM vs CPPO", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig2_sam_ablation.png")
    return m_sam


# ─── Experiment 3: SAC algorithm comparison ───────────────────────────────────

def exp3_sac_comparison(trade_llm, unique_dates, pvs_cppo, bh):
    sac_path = "trained_models/agent_sac_llm_300_ep.pth"
    if not os.path.exists(sac_path):
        print(f"\n[Exp 3] SAC model not found, skipping ({sac_path})")
        return None
    print("\n[Exp 3] Algorithm comparison: SAC vs CPPO (same LLM signals) …")
    from backtest_sac import load_sac_model
    n   = len(unique_dates)
    sac = load_sac_model(sac_path)

    env = StockTradingEnv(df=trade_llm, stock_dim=STOCK_DIM, **ENV_KWARGS)
    obs  = np.array(env.reset()[0], dtype=np.float32)
    pvs  = [INITIAL_AMOUNT]; done = False
    while not done:
        a = sac.act(torch.FloatTensor(obs))
        obs, _, term, trunc, _ = env.step(a)
        obs  = np.array(obs, dtype=np.float32); done = term or trunc
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])
    pvs_sac = pvs[:n]

    m_sac  = compute_full_metrics(pvs_sac,  bh, name="SAC")
    m_cppo = compute_full_metrics(pvs_cppo, bh, name="CPPO")
    pd.DataFrame([m_cppo, m_sac]).to_csv("backtest_results/algorithm_comparison.csv", index=False)
    print(f"  CPPO Sharpe={m_cppo['sharpe_ratio']:.4f}  SAC Sharpe={m_sac['sharpe_ratio']:.4f}")
    return m_sac


# ─── Experiment 4: Multi-seed stats ───────────────────────────────────────────

def exp4_multi_seed(bh, unique_dates):
    n    = len(unique_dates)
    seed_dir = "trained_models/seeds"
    seed_paths = sorted([
        os.path.join(seed_dir, f)
        for f in os.listdir(seed_dir)
        if f.startswith("agent_seed") and f.endswith(".pth")
    ])
    if not seed_paths:
        print("\n[Exp 4] No seed models found, skipping.")
        return None
    print(f"\n[Exp 4] Multi-seed stats ({len(seed_paths)} seeds) …")
    all_pvs     = []
    all_sharpes = []
    for sp in seed_paths:
        model = load_cppo_model(sp, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)
        trade, dates = load_trade("llm")
        pvs   = run_cppo(model, trade, n)
        m     = compute_full_metrics(pvs, bh, name=os.path.basename(sp))
        all_pvs.append(pvs)
        all_sharpes.append(m["sharpe_ratio"])
        print(f"  {os.path.basename(sp)}: Sharpe={m['sharpe_ratio']:.4f}")

    if len(all_sharpes) >= 2:
        from scipy.stats import wilcoxon
        bh_sharpes = [compute_full_metrics(bh, bh)["sharpe_ratio"]] * len(all_sharpes)
        try:
            stat, pval = wilcoxon(all_sharpes, bh_sharpes)
        except Exception:
            pval = float("nan")
        print(f"  Mean Sharpe: {np.mean(all_sharpes):.4f} ± {np.std(all_sharpes):.4f}")
        print(f"  Wilcoxon p-value vs B&H: {pval:.4f}")

    pd.DataFrame({
        "model": [os.path.basename(sp) for sp in seed_paths],
        "sharpe": all_sharpes,
    }).to_csv("backtest_results/multi_seed_sharpes.csv", index=False)

    dates_dt = pd.to_datetime(unique_dates[:n])
    fig, ax  = plt.subplots(figsize=(12,5))
    pv_arr   = np.array(all_pvs)
    mean_pv  = pv_arr.mean(axis=0)
    std_pv   = pv_arr.std(axis=0)
    ax.plot(dates_dt, mean_pv/mean_pv[0], "#2563EB", lw=2.2, label="Mean across seeds")
    ax.fill_between(dates_dt,
                    (mean_pv - std_pv)/mean_pv[0], (mean_pv + std_pv)/mean_pv[0],
                    alpha=0.2, color="#2563EB", label="±1 std")
    ax.plot(dates_dt, bh[:n]/bh[0], "#94A3B8", lw=1.5, ls=":", label="B&H")
    ax.set_title(f"Multi-Seed Robustness ({len(all_pvs)} seeds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig3_multi_seed.png")
    return all_sharpes


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_sam",  action="store_true")
    parser.add_argument("--skip_sac",  action="store_true")
    parser.add_argument("--skip_oos",  action="store_true")
    parser.add_argument("--skip_seeds",action="store_true")
    args = parser.parse_args()

    print("=== Full Evaluation Pipeline ===")
    print(f"State space: {STATE_SPACE}  |  Indicators: {len(INDICATORS)}  |  Stocks: {STOCK_DIM}")

    trade_llm,     unique_dates = load_trade("llm")
    trade_neutral, _            = load_trade("neutral")
    n  = len(unique_dates)
    bh = bh_series(trade_llm, n)

    # Exp 1: Main result
    m_llm, m_neutral, m_bh, pvs_cppo, bh, unique_dates = \
        exp1_main_comparison(trade_llm, trade_neutral, unique_dates)

    # Exp 2: SAM ablation
    if not args.skip_sam:
        exp2_sam_ablation(trade_llm, unique_dates, pvs_cppo, bh)

    # Exp 3: SAC comparison
    if not args.skip_sac:
        exp3_sac_comparison(trade_llm, unique_dates, pvs_cppo, bh)

    # Exp 4: Multi-seed
    if not args.skip_seeds:
        exp4_multi_seed(bh, unique_dates)

    # Print summary table
    print(f"\n{'═'*70}")
    print("  Summary Table (2019-2023 test period)")
    print(f"{'═'*70}")
    rows = [m_llm, m_neutral, m_bh]
    labels = ["CPPO + LLM Signals", "CPPO + Neutral", "Buy & Hold (EW)"]
    for lbl, m in zip(labels, rows):
        print(f"  {lbl:<24}: "
              f"CR={m['cumulative_return']:>7.2f}%  "
              f"Sharpe={m['sharpe_ratio']:.4f}  "
              f"MDD={m['max_drawdown_pct']:>6.2f}%  "
              f"Sortino={m['sortino_ratio']:.4f}")

    # Save final metrics JSON
    results = {
        "cppo_llm":     m_llm,
        "cppo_neutral": m_neutral,
        "buy_hold":     m_bh,
    }
    with open("backtest_results/full_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nFull results saved → backtest_results/full_eval_results.json")


if __name__ == "__main__":
    main()
