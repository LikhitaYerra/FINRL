#!/usr/bin/env python3
"""
Paired statistical diagnostics for semantic portfolio comparisons.

The RL seed table already reports seed-resampled intervals for policy training.
This script adds a complementary daily-return view for the deterministic
non-RL semantic baselines: SFP vs sentiment-only, SCW vs SFP, and supervised
semantic tilt vs price-only forecasting.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats


def returns_by_strategy(curves: pd.DataFrame, value_col: str = "portfolio_value") -> pd.DataFrame:
    wide = curves.pivot(index="date", columns="strategy", values=value_col).sort_index()
    return wide.pct_change().dropna()


def block_bootstrap_mean_ci(
    x: np.ndarray,
    *,
    block: int = 20,
    n_boot: int = 5000,
    seed: int = 7,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    starts = np.arange(max(1, n - block + 1))
    means = np.empty(n_boot)
    for i in range(n_boot):
        pieces = []
        while sum(len(p) for p in pieces) < n:
            start = int(rng.choice(starts))
            pieces.append(x[start:start + block])
        sample = np.concatenate(pieces)[:n]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def comparison_row(
    returns: pd.DataFrame,
    *,
    left: str,
    right: str,
    label: str,
) -> dict:
    aligned = returns[[left, right]].dropna()
    diff = aligned[left].to_numpy() - aligned[right].to_numpy()
    mean_bp = diff.mean() * 10_000
    ci_lo, ci_hi = block_bootstrap_mean_ci(diff)
    try:
        stat, pval = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        stat, pval = 0.0, 1.0
    sharpe_left = aligned[left].mean() / (aligned[left].std() + 1e-12) * np.sqrt(252)
    sharpe_right = aligned[right].mean() / (aligned[right].std() + 1e-12) * np.sqrt(252)
    win_rate = float((diff > 0).mean() * 100)
    return {
        "Comparison": label,
        "Mean active bp/day": mean_bp,
        "Bootstrap 95% CI low": ci_lo * 10_000,
        "Bootstrap 95% CI high": ci_hi * 10_000,
        "Sharpe delta": float(sharpe_left - sharpe_right),
        "Daily win rate (%)": win_rate,
        "Wilcoxon p": float(pval),
    }


def write_latex_table(rows: list[dict], out_path: str) -> None:
    body = []
    for row in rows:
        body.append(
            "{} & {:.3f} & [{:.3f}, {:.3f}] & {:+.3f} & {:.1f} & {:.3f} \\\\".format(
                row["Comparison"].replace("&", r"\&"),
                row["Mean active bp/day"],
                row["Bootstrap 95% CI low"],
                row["Bootstrap 95% CI high"],
                row["Sharpe delta"],
                row["Daily win rate (%)"],
                row["Wilcoxon p"],
            )
        )
    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired daily-return diagnostics: semantic baselines plus multi-seed RL vs EW benchmark. The confidence interval is a 20-trading-day block bootstrap over mean paired daily active returns, in basis points per day. Wilcoxon tests are paired over daily returns and are reported as diagnostics rather than definitive multiple-comparison-adjusted claims.}",
        r"\label{tab:semantic_stat_tests}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Mean bp/day} & \textbf{95\% CI} & \textbf{$\Delta$ Sharpe} & \textbf{Win \%} & \textbf{Wilcoxon $p$} \\",
        r"\midrule",
        "\n".join(body),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main() -> int:
    os.makedirs("backtest_results", exist_ok=True)
    semantic = pd.read_csv("backtest_results/semantic_factor_portfolio.csv")
    supervised = pd.read_csv("backtest_results/supervised_forecasting_curves.csv")
    semantic_ret = returns_by_strategy(semantic)
    supervised_ret = returns_by_strategy(supervised)

    rows = [
        comparison_row(
            semantic_ret,
            left="SFP (4 factors)",
            right="SFP (sentiment only)",
            label="SFP 4-factor vs sentiment-only",
        ),
        comparison_row(
            semantic_ret,
            left="SCW (conviction-weighted)",
            right="SFP (4 factors)",
            label="SCW vs equal-weight SFP",
        ),
        comparison_row(
            supervised_ret,
            left="Supervised price + semantic tilt",
            right="Supervised price-only",
            label="Semantic tilt vs price-only forecaster",
        ),
    ]

    ms_path = "backtest_results/multi_seed_portfolio.csv"
    if os.path.isfile(ms_path):
        ms = pd.read_csv(ms_path, parse_dates=["date"]).sort_values("date")
        ms = ms.drop_duplicates("date", keep="last")
        idx = ms["date"]
        ms_ret = pd.DataFrame(
            {
                "Multi-seed mean": ms["mean"].pct_change().to_numpy(),
                "EW buy-and-hold": ms["buy_hold"].pct_change().to_numpy(),
            },
            index=idx,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            comparison_row(
                ms_ret,
                left="Multi-seed mean",
                right="EW buy-and-hold",
                label="Multi-seed DP-PPO mean vs EW buy-and-hold",
            )
        )
    pd.DataFrame(rows).to_csv("backtest_results/semantic_stat_tests.csv", index=False)
    write_latex_table(rows, "paper/table_semantic_stat_tests.tex")
    for row in rows:
        print(
            f"{row['Comparison']}: mean={row['Mean active bp/day']:.3f} bp/day, "
            f"CI=[{row['Bootstrap 95% CI low']:.3f}, {row['Bootstrap 95% CI high']:.3f}], "
            f"dSharpe={row['Sharpe delta']:+.3f}, p={row['Wilcoxon p']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
