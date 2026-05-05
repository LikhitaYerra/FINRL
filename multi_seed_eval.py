"""
Multi-seed evaluation harness.

Runs the DP-PPO agent across N random seeds, then reports:
  - mean ± std of key metrics (full-signal vs neutral-masked evaluation per seed)
  - Bootstrap 95% CI on the mean Sharpe / mean cumulative return across seeds
  - Per-seed portfolio curves for dashboards / plots

Since retraining is sequential per seed, use ``--seeds`` for how many checkpoints
(0..N-1) to train or evaluate. Rough guide: 30 epochs × 4 MPI workers ≈ under 1 h/seed
on a typical workstation (varies a lot); 16 seeds is often an overnight batch.

Two modes:
  1. --mode eval   : Load existing checkpoints from ``--out_dir`` (default trained_models/seeds)
  2. --mode train  : Train all seeds sequentially then evaluate

Also emits ``paper/table_seed_robustness.tex`` when at least two seeds are evaluated.

Usage:
    python multi_seed_eval.py --mode train --seeds 16 --epochs 30
    python multi_seed_eval.py --mode train --seed-start 10 --seeds 11 --epochs 30   # seeds 10..20
    python multi_seed_eval.py --mode eval --seeds 16
    python multi_seed_eval.py --mode eval --seed-start 10 --seeds 11
    python multi_seed_eval.py --mode eval --seeds 16 --bootstrap-samples 20000
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv

INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_AMOUNT = 1_000_000
STOCK_DIM      = 30
STATE_SPACE    = 1 + 2 * STOCK_DIM + (len(INDICATORS) + 4) * STOCK_DIM  # 421
ACTION_SPACE   = STOCK_DIM

ENV_KWARGS = dict(
    hmax=100, initial_amount=INITIAL_AMOUNT,
    num_stock_shares=[0]*STOCK_DIM,
    buy_cost_pct=[0.001]*STOCK_DIM, sell_cost_pct=[0.001]*STOCK_DIM,
    reward_scaling=1e-4, state_space=STATE_SPACE, action_space=ACTION_SPACE,
    tech_indicator_list=INDICATORS, turbulence_threshold=380,
    drawdown_penalty=0.1, make_plots=False, print_verbosity=999,
)

LLM_SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
NEUTRAL_SCORE = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Run one backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    model_path: str,
    trade_df: pd.DataFrame,
    unique_dates: list,
    *,
    neutral_eval: bool = False,
) -> list:
    df = trade_df.copy()
    if neutral_eval:
        for col in LLM_SIGNAL_COLS:
            if col in df.columns:
                df[col] = NEUTRAL_SCORE

    env = StockTradingEnv(df=df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    ac  = load_cppo_model(model_path, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

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


# ─────────────────────────────────────────────────────────────────────────────
# Buy-and-hold baseline
# ─────────────────────────────────────────────────────────────────────────────

def bh_series(trade_df: pd.DataFrame) -> np.ndarray:
    avg = trade_df.groupby("date")["close"].mean().sort_index()
    return (INITIAL_AMOUNT * avg.values / avg.values[0])


# ─────────────────────────────────────────────────────────────────────────────
# Train one seed  (calls mpirun as a subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def train_seed(seed: int, epochs: int, n_cpu: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"agent_seed{seed}.pth")
    log_path   = os.path.join(out_dir, f"train_seed{seed}.log")

    if os.path.exists(model_path):
        print(f"  Seed {seed}: checkpoint already exists, skipping training.")
        return model_path

    # Temporarily override the default save path via env var
    cmd = [
        "mpirun", "-np", str(n_cpu),
        sys.executable, "train_cppo_multi_signal.py",
        "--local_data", "train_data_multi_signal_2013_2018.csv",
        "--epochs", str(epochs),
        "--cpu", str(n_cpu),
        "--seed", str(seed),
        "--exp_name", f"cppo_seed{seed}",
    ]
    print(f"  Seed {seed}: training ({epochs} epochs, {n_cpu} workers)…")
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)

    # The script saves to trained_models/agent_cppo_multi_signal_{epochs}_epochs.pth
    default_path = f"trained_models/agent_cppo_multi_signal_{epochs}_epochs.pth"
    if os.path.exists(default_path):
        import shutil
        shutil.copy(default_path, model_path)
        print(f"  Seed {seed}: saved to {model_path}")
    else:
        print(f"  Seed {seed}: WARNING — model file not found at {default_path}")

    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean ± std across seeds for every numeric key."""
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float, bool))]
    result = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if m[k] is not None]
        result[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
            "min":  round(float(np.min(vals)),  4),
            "max":  round(float(np.max(vals)),  4),
        }
    return result


def paired_full_vs_neutral_tests(
    all_metrics: list[dict],
    all_metrics_neutral: list[dict],
) -> dict:
    """Paired seed-level tests comparing full scoring to neutral-masked eval."""
    tests = {}
    metric_names = ["cumulative_return", "sharpe_ratio", "max_drawdown_pct"]
    for metric in metric_names:
        full = np.array([float(m[metric]) for m in all_metrics], dtype=float)
        neutral = np.array([float(m[metric]) for m in all_metrics_neutral], dtype=float)
        diff = full - neutral
        row = {
            "n": int(len(diff)),
            "mean_delta": round(float(np.mean(diff)), 4),
            "std_delta": round(float(np.std(diff, ddof=1)), 4) if len(diff) > 1 else 0.0,
            "per_seed_delta": [round(float(x), 4) for x in diff],
        }
        if len(diff) >= 2:
            t_res = stats.ttest_rel(full, neutral, nan_policy="omit")
            row["paired_t_stat"] = round(float(t_res.statistic), 4)
            row["paired_t_pval"] = round(float(t_res.pvalue), 4)
            try:
                w_res = stats.wilcoxon(full, neutral, alternative="two-sided")
                row["wilcoxon_stat"] = round(float(w_res.statistic), 4)
                row["wilcoxon_pval"] = round(float(w_res.pvalue), 4)
            except ValueError:
                row["wilcoxon_stat"] = None
                row["wilcoxon_pval"] = None
        tests[metric] = row
    return tests


def write_seed_robustness_tex(
    n_seeds: int,
    agg_full: dict,
    agg_neutral: dict,
    out_path: str = "paper/table_seed_robustness.tex",
    bootstrap_ci=None,
    paired_tests=None,
) -> None:
    """Paper table: mean ± std for CR and Sharpe (full vs neutral-masked eval)."""
    if n_seeds < 2:
        return

    def pm(stats: dict, ndm: int, nds: int) -> str:
        return f"${stats['mean']:.{ndm}f} \\pm {stats['std']:.{nds}f}$"

    cr_f = agg_full["cumulative_return"]
    sh_f = agg_full["sharpe_ratio"]
    cr_n = agg_neutral["cumulative_return"]
    sh_n = agg_neutral["sharpe_ratio"]

    caption = (
        "Multi-seed robustness (%d seeds, 2019--2023). Mean $\\pm$ std across seeds for cumulative "
        "return and Sharpe. \\textit{Neutral eval} masks all LLM coordinates to neutral using the same checkpoints."
    ) % n_seeds

    latex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:seed_robust}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Variant} & \\textbf{CR (\\%)} & \\textbf{Sharpe} \\\\",
        "\\midrule",
        f"DP-PPO (full signals) & {pm(cr_f, 2, 2)} & {pm(sh_f, 3, 3)} \\\\",
        f"DP-PPO (neutral eval) & {pm(cr_n, 2, 2)} & {pm(sh_n, 3, 3)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\n  LaTeX seed robustness → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["train","eval"], default="eval")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed index (inclusive). Training/eval visits seed_start, seed_start+1, … for --seeds steps.",
    )
    parser.add_argument("--seeds",   type=int, default=16,
                        help="How many consecutive seeds to run (default 16). With --seed-start 0 → 0..15.")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--cpu",     type=int, default=4)
    parser.add_argument("--out_dir", default="trained_models/seeds")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20_000,
        help="Bootstrap resamples for 95%% CI on mean Sharpe/CR across seeds (default 20000).",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        print("ERROR: --seeds must be >= 1")
        return
    seed_lo = args.seed_start
    seed_hi = args.seed_start + args.seeds  # exclusive upper bound for range()

    os.makedirs("backtest_results", exist_ok=True)

    print(f"Loading trade data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)

    bh = bh_series(trade)[:len(unique_dates)]

    # ── Train phase ──────────────────────────────────────────────────────────
    if args.mode == "train":
        print(f"\nTraining seeds {seed_lo}..{seed_hi - 1} ({args.seeds} runs) × {args.epochs} epochs …")
        for seed in range(seed_lo, seed_hi):
            train_seed(seed, args.epochs, args.cpu, args.out_dir)

    # ── Eval phase ───────────────────────────────────────────────────────────
    print(f"\nEvaluating seeds …")

    # Collect available models
    seed_paths = []
    for seed in range(seed_lo, seed_hi):
        p = os.path.join(args.out_dir, f"agent_seed{seed}.pth")
        if os.path.exists(p):
            seed_paths.append((seed, p))

    # Also include the main pre-trained model as seed 0 if no seed models
    if not seed_paths:
        main_model = "trained_models/agent_cppo_multi_signal_30_epochs.pth"
        if os.path.exists(main_model):
            print(f"  No seed models found — using main model as seed 0")
            seed_paths = [(0, main_model)]

    if not seed_paths:
        print("ERROR: No trained models found. Run with --mode train first.")
        return

    # Run each seed (full signals + neutral-masked evaluation of same weights)
    all_pvs           = []
    all_pvs_neutral   = []
    all_metrics       = []
    all_metrics_neutral = []
    for seed, path in seed_paths:
        print(f"  Seed {seed}: {path}")
        pvs = run_one(path, trade, unique_dates, neutral_eval=False)
        pvn = run_one(path, trade, unique_dates, neutral_eval=True)
        all_pvs.append(pvs)
        all_pvs_neutral.append(pvn)
        m = compute_full_metrics(pvs, bh, name=f"DP-PPO seed={seed} (full)")
        mn = compute_full_metrics(pvn, bh, name=f"DP-PPO seed={seed} (neutral eval)")
        all_metrics.append(m)
        all_metrics_neutral.append(mn)
        print(
            f"    full: CR={m['cumulative_return']:.2f}%  SR={m['sharpe_ratio']:.4f}  "
            f"neutral-eval: CR={mn['cumulative_return']:.2f}%  SR={mn['sharpe_ratio']:.4f}"
        )

    # Aggregate
    agg = aggregate_metrics(all_metrics)
    agg_neutral = aggregate_metrics(all_metrics_neutral)
    print(f"\n{'═'*55}")
    print(f"  Aggregated over {len(seed_paths)} seed(s) — full-signal evaluation")
    print(f"{'═'*55}")
    for metric, stats_blob in agg.items():
        if metric in ["cumulative_return","annual_return","sharpe_ratio","sortino_ratio",
                      "rachev_ratio","max_drawdown_pct","calmar_ratio","outperf_overall","wilcoxon_pval"]:
            print(f"  {metric:<26}: {stats_blob['mean']:.4f} ± {stats_blob['std']:.4f}")

    print(f"\n{'═'*55}")
    print(f"  Aggregated — neutral-masked evaluation (same checkpoints)")
    print(f"{'═'*55}")
    for metric, stats_blob in agg_neutral.items():
        if metric in ["cumulative_return","annual_return","sharpe_ratio","sortino_ratio",
                      "rachev_ratio","max_drawdown_pct","calmar_ratio","outperf_overall","wilcoxon_pval"]:
            print(f"  {metric:<26}: {stats_blob['mean']:.4f} ± {stats_blob['std']:.4f}")

    paired_tests = paired_full_vs_neutral_tests(all_metrics, all_metrics_neutral)
    print(f"\n{'═'*55}")
    print("  Paired seed-level tests — full scoring vs neutral eval")
    print(f"{'═'*55}")
    for metric in ["cumulative_return", "sharpe_ratio", "max_drawdown_pct"]:
        row = paired_tests[metric]
        print(
            f"  {metric:<26}: Δ={row['mean_delta']:.4f}  "
            f"t-p={row.get('paired_t_pval', float('nan')):.4f}  "
            f"wilcoxon-p={row.get('wilcoxon_pval', float('nan')):.4f}"
        )

    # ── Bootstrap 95% CI on seed Sharpe / cumulative-return distribution ────────
    bootstrap_ci: dict = {}
    sharpe_vals = np.array([float(m["sharpe_ratio"]) for m in all_metrics])
    cr_vals = np.array([float(m["cumulative_return"]) for m in all_metrics])
    sharpe_vals_n = np.array([float(m["sharpe_ratio"]) for m in all_metrics_neutral])
    cr_vals_n = np.array([float(m["cumulative_return"]) for m in all_metrics_neutral])
    if len(sharpe_vals) >= 3:
        rng = np.random.default_rng(0)
        n_b = max(1000, int(args.bootstrap_samples))
        idx_draw = rng.integers(0, len(sharpe_vals), size=(n_b, len(sharpe_vals)))
        boot_sh = sharpe_vals[idx_draw].mean(axis=1)
        boot_cr = cr_vals[idx_draw].mean(axis=1)
        bootstrap_ci["sharpe_mean_95ci"] = [
            round(float(np.percentile(boot_sh, 2.5)), 4),
            round(float(np.percentile(boot_sh, 97.5)), 4),
        ]
        bootstrap_ci["cumulative_return_mean_95ci"] = [
            round(float(np.percentile(boot_cr, 2.5)), 4),
            round(float(np.percentile(boot_cr, 97.5)), 4),
        ]
        boot_shn = sharpe_vals_n[idx_draw].mean(axis=1)
        boot_crn = cr_vals_n[idx_draw].mean(axis=1)
        bootstrap_ci["sharpe_neutral_mean_95ci"] = [
            round(float(np.percentile(boot_shn, 2.5)), 4),
            round(float(np.percentile(boot_shn, 97.5)), 4),
        ]
        bootstrap_ci["cumulative_return_neutral_mean_95ci"] = [
            round(float(np.percentile(boot_crn, 2.5)), 4),
            round(float(np.percentile(boot_crn, 97.5)), 4),
        ]
        print(f"\n  Bootstrap 95% CI (mean across seeds; seeds resampled with replacement):")
        print(f"    Sharpe mean [full]: [{bootstrap_ci['sharpe_mean_95ci'][0]}, {bootstrap_ci['sharpe_mean_95ci'][1]}]")
        print(f"    CR mean (%) [full]: [{bootstrap_ci['cumulative_return_mean_95ci'][0]}, {bootstrap_ci['cumulative_return_mean_95ci'][1]}]")
        print(f"    Sharpe mean [neutral eval]: [{bootstrap_ci['sharpe_neutral_mean_95ci'][0]}, {bootstrap_ci['sharpe_neutral_mean_95ci'][1]}]")
        print(f"    CR mean (%) [neutral eval]: [{bootstrap_ci['cumulative_return_neutral_mean_95ci'][0]}, {bootstrap_ci['cumulative_return_neutral_mean_95ci'][1]}]")

    # Save
    result = {
        "n_seeds": len(seed_paths),
        "per_seed": all_metrics,
        "per_seed_neutral_eval": all_metrics_neutral,
        "aggregated": agg,
        "aggregated_neutral_eval": agg_neutral,
        "bootstrap_ci_seeds": bootstrap_ci if bootstrap_ci else {},
        "paired_full_vs_neutral_eval": paired_tests,
        "dates": unique_dates,
        "portfolio_values": [pvs for pvs in all_pvs],
        "portfolio_values_neutral_eval": [pvs for pvs in all_pvs_neutral],
        "buy_hold": bh.tolist(),
    }
    out_path = "backtest_results/multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_path}")

    write_seed_robustness_tex(
        len(seed_paths),
        agg,
        agg_neutral,
        bootstrap_ci=bootstrap_ci if bootstrap_ci else None,
        paired_tests=paired_tests,
    )

    # Portfolio CSV for dashboard
    df_out = pd.DataFrame({"date": unique_dates})
    for i, (seed, _) in enumerate(seed_paths):
        df_out[f"seed_{seed}"] = all_pvs[i]
    df_out["buy_hold"] = bh[:len(unique_dates)]
    if len(all_pvs) > 1:
        stacked = np.array(all_pvs)
        df_out["mean"] = stacked.mean(axis=0)
        df_out["std"]  = stacked.std(axis=0)
    df_out.to_csv("backtest_results/multi_seed_portfolio.csv", index=False)
    print(f"  Saved → backtest_results/multi_seed_portfolio.csv")


if __name__ == "__main__":
    main()
