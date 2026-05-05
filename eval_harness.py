"""
Unified Evaluation Harness

Runs ALL experiments in a single script and produces the exact tables
and figures needed for the paper:

  TABLE 1 — Strategy comparison (main results)
  TABLE 2 — Signal ablation study (optional sentiment-only trained row when checkpoint exists)
  TABLE 3 — Multi-seed DP-PPO mean±std (when ``backtest_results/multi_seed_results.json`` exists)
  TABLE seed robustness — mean±std full vs neutral-eval + bootstrap CI (same JSON, ≥2 seeds)
  FIGURE 1 — Equity curves + confidence band (multi-seed)
  FIGURE 2 — Signal ablation bar chart
  FIGURE 3 — Regime overlay
  FIGURE 4 — Attention weight heatmap over time

Usage:
    python eval_harness.py                        # full evaluation
    python eval_harness.py --scalar_model PATH    # include sentiment-only checkpoint
    python eval_harness.py --skip_regime          # skip HMM fitting
    python eval_harness.py --skip_ablation        # skip ablation
"""

from __future__ import annotations

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
from scipy import stats

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv

INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]


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

SIGNAL_NAMES = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
# Sentiment-only policies are trained with non-sentiment axes fixed at neutral; match at test.
SCALAR_EVAL_MASK = ["llm_risk", "llm_confidence", "llm_volatility_forecast"]
DEFAULT_SCALAR_MODEL_PATHS = (
    "trained_models/agent_cppo_multi_signal_30_epochs_scalar_sentiment.pth",
    "trained_models/agent_cppo_multi_signal_100_epochs_scalar_sentiment.pth",
)
NEUTRAL = 3.0

FIG_DIR  = "paper/figures"
DATA_DIR = "backtest_results"


# ─── Core run ─────────────────────────────────────────────────────────────────

def run_agent(ac, trade_df, unique_dates, signal_mask=None, action_scale=1.0):
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
        action = ac.act(torch.FloatTensor(obs)) * action_scale
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])

    return pvs[:len(unique_dates)]


# ─── FIGURE 1 — Main equity curves with confidence band ─────────────────────

def figure_main_equity(results: dict, dates: list, out: str):
    dates_dt = pd.to_datetime(dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    bh_pv = np.array(results["Buy & Hold (EW)"])

    for name, color, lw, ls in [
        ("DP-PPO (all signals)", "#2563EB", 2.4, "-"),
        ("Buy & Hold (EW)",    "#94A3B8", 1.8, "--"),
        ("Momentum (top-10)",  "#F59E0B", 1.5, "-"),
    ]:
        if name not in results:
            continue
        pv   = np.array(results[name])
        norm = pv / pv[0]
        n    = min(len(norm), len(dates_dt))
        ax1.plot(dates_dt[:n], norm[:n], label=name, color=color, lw=lw, ls=ls)

    # Multi-seed confidence band (if available)
    ms_path = f"{DATA_DIR}/multi_seed_portfolio.csv"
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path)
        seed_cols = [c for c in ms.columns if c.startswith("seed_")]
        if len(seed_cols) >= 2:
            stacked = ms[seed_cols].values
            m_val   = stacked.mean(axis=1) / stacked[0].mean()
            s_val   = stacked.std(axis=1)  / stacked[0].mean()
            dt_band = pd.to_datetime(ms["date"].values)
            ax1.fill_between(dt_band, m_val - s_val, m_val + s_val,
                             alpha=0.15, color="#2563EB", label="±1σ (multi-seed)")

    ax1.set_ylabel("Normalised Portfolio Value (start=1)", fontsize=11)
    ax1.set_title("LLM-Enhanced DP-PPO vs Baselines — NASDAQ-100 (2019–2023)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    # Drawdown panel
    pv_agent = np.array(results["DP-PPO (all signals)"])
    peak     = np.maximum.accumulate(pv_agent)
    dd       = (pv_agent - peak) / peak * 100
    n_dd     = min(len(dd), len(dates_dt))
    ax2.fill_between(dates_dt[:n_dd], dd[:n_dd], 0, color="#2563EB", alpha=0.4, label="DP-PPO drawdown")
    ax2.plot(dates_dt[:n_dd], dd[:n_dd], color="#2563EB", lw=0.8)
    bh_dd = (bh_pv - np.maximum.accumulate(bh_pv)) / np.maximum.accumulate(bh_pv) * 100
    n_bh  = min(len(bh_dd), len(dates_dt))
    ax2.plot(dates_dt[:n_bh], bh_dd[:n_bh], color="#94A3B8", lw=0.8, ls="--", label="B&H drawdown")
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 1 → {out}")


# ─── FIGURE 2 — Ablation bar chart ───────────────────────────────────────────

def figure_ablation_bars(df_ablation: pd.DataFrame, out: str):
    strategies = [s for s in df_ablation["Strategy"] if "DP-PPO" in s]
    df = df_ablation[df_ablation["Strategy"].isin(strategies)].copy()
    df = df.sort_values("Sharpe", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#2563EB" if "all" in s else "#94A3B8" for s in df["Strategy"]]

    for ax, metric, label in zip(axes,
        ["CR (%)", "Sharpe", "Rachev"],
        ["Cumulative Return (%)", "Sharpe Ratio", "Rachev Ratio"]):
        bars = ax.barh(df["Strategy"], df[metric], color=colors)
        ax.set_xlabel(label)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        # Add value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.01 * abs(val), bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8)

    fig.suptitle("Signal Ablation Study — Impact of Removing Each LLM Signal",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 2 → {out}")


# ─── FIGURE 3 — Regime breakdown bar chart ───────────────────────────────────

def figure_regime_breakdown(regime_csv: str, out: str):
    if not os.path.exists(regime_csv):
        print(f"  Skipping FIGURE 3 (no regime_results.csv)")
        return

    df = pd.read_csv(regime_csv)
    for col in ["cppo_plain", "cppo_regime", "buy_hold"]:
        df[f"ret_{col}"] = df[col].pct_change()

    breakdown = df.groupby("regime").agg(
        cppo_mean=("ret_cppo_regime", "mean"),
        plain_mean=("ret_cppo_plain", "mean"),
        bh_mean=("ret_buy_hold", "mean"),
        n=("regime", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    x    = np.arange(len(breakdown))
    w    = 0.25
    ax.bar(x - w, breakdown["plain_mean"]*100, w, label="DP-PPO (plain)", color="#94A3B8")
    ax.bar(x,     breakdown["cppo_mean"]*100,  w, label="DP-PPO + Regime", color="#2563EB")
    ax.bar(x + w, breakdown["bh_mean"]*100,    w, label="Buy & Hold", color="#F59E0B")
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown["regime"])
    ax.set_ylabel("Mean Daily Return (%)")
    ax.set_title("Per-Regime Daily Return Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 3 → {out}")


# ─── TABLE helpers ────────────────────────────────────────────────────────────

def _latex_escape_header(name: str) -> str:
    """Escape % and unescaped underscores in table headers for LaTeX."""
    return name.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_")


def _latex_escape_cell(val) -> str:
    if isinstance(val, float):
        return f"{val:.3f}"
    s = str(val)
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("_", "\\_")


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    ncols = "l" + "r" * (len(cols) - 1)
    header = " & ".join([f"\\textbf{{{_latex_escape_header(c)}}}" for c in cols]) + " \\\\"
    top = "\\toprule"
    mid = "\\midrule"
    bot = "\\bottomrule"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(_latex_escape_cell(val) for val in row) + " \\\\")

    tabular = "\n".join([
        f"\\begin{{tabular}}{{{ncols}}}",
        top, header, mid,
        "\n".join(rows),
        bot,
        "\\end{tabular}",
    ])
    if len(cols) >= 7:
        tabular = "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}"

    return "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        tabular,
        "\\end{table}",
    ])


def write_multiseed_latex(json_path: str, out_path: str) -> bool:
    """
    Emit ``paper/table_multiseed.tex`` from ``multi_seed_eval.py`` JSON output.
    Returns True if the file was written.
    """
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    agg = data.get("aggregated") or {}
    n_seeds = int(data.get("n_seeds", 0))
    if not agg or n_seeds < 2:
        return False

    metric_rows = [
        ("cumulative_return", "Cumulative return (\\%)"),
        ("annual_return", "Annual return (\\%)"),
        ("sharpe_ratio", "Sharpe ratio"),
        ("sortino_ratio", "Sortino ratio"),
        ("max_drawdown_pct", "Max drawdown (\\%)"),
        ("rachev_ratio", "Rachev ratio"),
        ("calmar_ratio", "Calmar ratio"),
        ("cvar_5pct", "CVaR-5\\%"),
    ]

    body = []
    for key, label in metric_rows:
        block = agg.get(key)
        if not isinstance(block, dict):
            continue
        mu = block.get("mean")
        sd = block.get("std")
        if mu is None or sd is None:
            continue
        body.append(f"{label} & {mu:.3f} & {sd:.3f} \\\\")

    if not body:
        return False

    latex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Multi-seed DP-PPO ({n_seeds} seeds, 2019--2023). Mean and standard deviation of test-window metrics.}}",
        "\\label{tab:multiseed}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} \\\\",
        "\\midrule",
        "\n".join(body),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    return True


def sync_seed_robustness_tex(json_path: str, out_path: str = "paper/table_seed_robustness.tex") -> bool:
    """Mirror ``multi_seed_eval.py`` seed table into LaTeX when JSON has ≥2 seeds."""
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    n_seeds = int(data.get("n_seeds", 0))
    agg = data.get("aggregated") or {}
    agg_neutral = data.get("aggregated_neutral_eval") or {}
    if n_seeds < 2 or not agg or not agg_neutral:
        return False
    try:
        from multi_seed_eval import write_seed_robustness_tex
    except ImportError:
        return False
    bi = data.get("bootstrap_ci_seeds") or {}
    paired = data.get("paired_full_vs_neutral_eval") or {}
    write_seed_robustness_tex(
        n_seeds,
        agg,
        agg_neutral,
        out_path=out_path,
        bootstrap_ci=bi if bi else None,
        paired_tests=paired if paired else None,
    )
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument(
        "--scalar_model",
        default=None,
        help="Sentiment-only trained DP-PPO weights "
             "(train_cppo_multi_signal.py --scalar_sentiment_only --save_suffix _scalar_sentiment). "
             "If omitted, common filenames under trained_models/ are tried.",
    )
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_regime",   action="store_true")
    args = parser.parse_args()

    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  Unified Evaluation Harness")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    for col in SIGNAL_NAMES:
        trade[col] = trade.get(col, NEUTRAL).fillna(NEUTRAL)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    n = len(unique_dates)

    # Buy-and-hold
    bh = INITIAL_AMOUNT * (
        trade.reset_index().groupby("date")["close"].mean().sort_index().values /
        trade.reset_index().groupby("date")["close"].mean().sort_index().values[0]
    )[:n]

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[2] Loading model: {args.model}")
    ac = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    results = {}

    # ── Main DP-PPO ────────────────────────────────────────────────────────────
    print("[3] Running DP-PPO (all signals) …")
    pvs = run_agent(ac, trade, unique_dates)
    results["DP-PPO (all signals)"] = pvs
    m_main = compute_full_metrics(pvs, bh, name="DP-PPO (all signals)")
    print(f"  CR={m_main['cumulative_return']:.2f}%  SR={m_main['sharpe_ratio']:.4f}"
          f"  Rachev={m_main['rachev_ratio']:.4f}  MDD={m_main['max_drawdown_pct']:.2f}%")

    scalar_candidates = []
    if args.scalar_model:
        scalar_candidates.append(args.scalar_model)
    scalar_candidates.extend(DEFAULT_SCALAR_MODEL_PATHS)
    scalar_path = next((p for p in scalar_candidates if p and os.path.isfile(p)), None)
    if scalar_path:
        print(f"[3b] DP-PPO (sentiment-only train) … {scalar_path}")
        ac_scalar = load_cppo_model(scalar_path, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)
        pvs_scalar = run_agent(ac_scalar, trade, unique_dates, signal_mask=SCALAR_EVAL_MASK)
        results["DP-PPO (sentiment-only train)"] = pvs_scalar
        m_sc = compute_full_metrics(pvs_scalar, bh, name="DP-PPO (sentiment-only train)")
        print(f"  CR={m_sc['cumulative_return']:.2f}%  SR={m_sc['sharpe_ratio']:.4f}"
              f"  Rachev={m_sc['rachev_ratio']:.4f}  MDD={m_sc['max_drawdown_pct']:.2f}%")
    else:
        print("[3b] Skipping sentiment-only baseline (no scalar checkpoint; train with "
              "--scalar_sentiment_only --save_suffix _scalar_sentiment)")

    # ── Ablation ──────────────────────────────────────────────────────────────
    if not args.skip_ablation:
        print("[4] Running signal ablation …")
        for sig in SIGNAL_NAMES:
            label = f"DP-PPO (no {sig.replace('llm_','')})"
            pv_abl = run_agent(ac, trade, unique_dates, signal_mask=[sig])
            results[label] = pv_abl

        pv_neutral = run_agent(ac, trade, unique_dates, signal_mask=SIGNAL_NAMES)
        results["DP-PPO (neutral)"] = pv_neutral

    # ── Momentum baseline ──────────────────────────────────────────────────
    print("[5] Running Momentum baseline …")
    from ablation import momentum_strategy, equal_volatility_strategy
    pvs_mom = momentum_strategy(trade.reset_index(), n_days_lookback=20, top_k=10)
    results["Momentum (top-10)"] = pvs_mom.tolist()

    pvs_ev = equal_volatility_strategy(trade.reset_index(), n)
    results["Equal-Vol (risk parity)"] = pvs_ev.tolist()
    results["Buy & Hold (EW)"] = bh.tolist()

    # ── Collect all metrics ────────────────────────────────────────────────
    print("\n[6] Computing metrics …")
    table_rows = []
    for name, pvs in results.items():
        m = compute_full_metrics(pvs, bh, name=name)
        table_rows.append({
            "Strategy":    name,
            "CR (%)":      m["cumulative_return"],
            "AR (%)":      m["annual_return"],
            "Sharpe":      m["sharpe_ratio"],
            "Sortino":     m["sortino_ratio"],
            "MDD (%)":     m["max_drawdown_pct"],
            "Rachev":      m["rachev_ratio"],
            "CVaR-5%":     m["cvar_5pct"],
            "Calmar":      m["calmar_ratio"],
            "Outperf (%)": m.get("outperf_overall", float("nan")),
        })

    df_results = pd.DataFrame(table_rows)
    df_results.to_csv(f"{DATA_DIR}/full_results_table.csv", index=False)

    print(f"\n{'═'*100}")
    print(df_results.to_string(index=False))

    # ── LaTeX tables ─────────────────────────────────────────────────────────
    os.makedirs("paper", exist_ok=True)

    # Table 1: main strategies only
    main_strats = ["DP-PPO (all signals)", "Buy & Hold (EW)", "Momentum (top-10)", "Equal-Vol (risk parity)"]
    df_t1 = df_results[df_results["Strategy"].isin(main_strats)][
        ["Strategy","CR (%)","Sharpe","Sortino","MDD (%)","Rachev","CVaR-5%","Calmar"]
    ]
    latex_t1 = render_latex_table(df_t1, "Main Strategy Comparison (2019–2023)", "tab:main_results")
    with open("paper/table_main.tex", "w") as f:
        f.write(latex_t1)
    print(f"\n  LaTeX TABLE 1 → paper/table_main.tex")

    # Table 2: ablation
    abl_strats = [k for k in df_results["Strategy"] if "DP-PPO" in k]
    df_t2 = df_results[df_results["Strategy"].isin(abl_strats)][
        ["Strategy","CR (%)","Sharpe","Rachev","MDD (%)"]
    ]
    latex_t2 = render_latex_table(df_t2, "Signal Ablation Study — DP-PPO Variants", "tab:ablation")
    with open("paper/table_ablation.tex", "w") as f:
        f.write(latex_t2)
    print(f"  LaTeX TABLE 2 → paper/table_ablation.tex")

    ms_json = f"{DATA_DIR}/multi_seed_results.json"
    if write_multiseed_latex(ms_json, "paper/table_multiseed.tex"):
        print(f"  LaTeX TABLE 3 (multi-seed) → paper/table_multiseed.tex")
    else:
        print(
            "  Skipping multi-seed LaTeX (run: python multi_seed_eval.py --mode eval --seeds 5)"
        )
    if sync_seed_robustness_tex(ms_json):
        print(f"  LaTeX seed robustness → paper/table_seed_robustness.tex")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\n[7] Generating figures …")
    figure_main_equity(results, unique_dates,
                       f"{FIG_DIR}/fig1_equity_curves.png")
    if not args.skip_ablation:
        figure_ablation_bars(df_results,
                             f"{FIG_DIR}/fig2_ablation.png")
    figure_regime_breakdown(f"{DATA_DIR}/regime_results.csv",
                            f"{FIG_DIR}/fig3_regime_breakdown.png")

    # ── Regime strategy ──────────────────────────────────────────────────────
    if not args.skip_regime:
        try:
            print("[8] Running regime-switching strategy …")
            from regime_strategy import RegimeDetector, build_market_features, run_regime_strategy
            train  = pd.read_csv("train_data_multi_signal_2013_2018.csv")
            t_feat = build_market_features(train, sorted(train["date"].unique()))
            r_feat = build_market_features(trade.reset_index(), unique_dates)
            det    = RegimeDetector(3)
            det.fit(t_feat)
            pvs_r, regimes = run_regime_strategy(ac, trade, unique_dates, det, r_feat)
            results["DP-PPO + Regime Switch"] = pvs_r
            m_r = compute_full_metrics(pvs_r, bh, name="DP-PPO + Regime Switch")
            print(f"  CR={m_r['cumulative_return']:.2f}%  SR={m_r['sharpe_ratio']:.4f}")
        except Exception as e:
            print(f"  Regime strategy error (install hmmlearn): {e}")

    # ── Wilcoxon significance test ────────────────────────────────────────
    print("\n[9] Statistical significance tests …")
    bh_ret    = np.diff(np.array(bh)) / bh[:-1]
    agent_ret = np.diff(np.array(results["DP-PPO (all signals)"])) / np.array(results["DP-PPO (all signals)"])[:-1]
    diff = agent_ret[:len(bh_ret)] - bh_ret[:len(agent_ret)]
    if len(diff) > 0:
        stat, pval = stats.wilcoxon(diff, alternative="two-sided")
        print(f"  Wilcoxon DP-PPO vs B&H: stat={stat:.2f}, p={pval:.4f} "
              f"({'SIGNIFICANT' if pval<0.05 else 'not significant'} at α=0.05)")

    print("\n" + "═" * 60)
    print("  Evaluation complete.")
    print("  All outputs in:")
    print(f"    {DATA_DIR}/  (CSVs)")
    print(f"    {FIG_DIR}/   (figures)")
    print(f"    paper/       (LaTeX: table_main, table_ablation; table_multiseed / table_seed_robustness if JSON)")
    print("═" * 60)


if __name__ == "__main__":
    main()
