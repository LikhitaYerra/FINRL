#!/usr/bin/env python3
"""
Signal-quality report: Spearman IC between LLM score panels and forward returns.

Uses the same pooled panel logic as ``lookahead_bias.compute_ic`` but runs offline on
existing CSVs (no API calls). Helps justify that textual coordinates carry incremental
predictive content versus pure noise.

Outputs:
  backtest_results/signal_ic_report.json
  paper/table_signal_ic.tex

Usage:
    python signal_ic_report.py
    python signal_ic_report.py --csv trade_data_multi_signal_2019_2023.csv --lag 5
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

SIGNAL_NAMES = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]


def compute_ic(df: pd.DataFrame, signal_col: str, lag: int = 5) -> dict:
    pivot_close = df.pivot(index="date", columns="tic", values="close").sort_index()
    pivot_sig = df.pivot(index="date", columns="tic", values=signal_col).sort_index().fillna(3.0)
    fwd_ret = pivot_close.pct_change(lag).shift(-lag)
    fwd_abs_ret = fwd_ret.abs()
    s = pivot_sig.values.flatten()
    r = fwd_ret.values.flatten()
    ar = fwd_abs_ret.values.flatten()
    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 50:
        return {
            "n": int(mask.sum()),
            "return_ic": float("nan"),
            "return_pval": float("nan"),
            "abs_return_ic": float("nan"),
            "abs_return_pval": float("nan"),
        }
    ic, pval = stats.spearmanr(s[mask], r[mask])
    mask_abs = np.isfinite(s) & np.isfinite(ar)
    abs_ic, abs_pval = stats.spearmanr(s[mask_abs], ar[mask_abs])
    return {
        "n": int(mask.sum()),
        "return_ic": float(ic),
        "return_pval": float(pval),
        "abs_return_ic": float(abs_ic),
        "abs_return_pval": float(abs_pval),
    }


def _latex_label(sig: str) -> str:
    return sig.replace("llm_", "").replace("_", r"\_")


def write_latex_table(results: dict, lag: int, out_path: str) -> None:
    rows = []
    for sig, blob in results.items():
        rows.append(
            "{} & {:.4f} & {:.2g} & {:.4f} & {:.2g} \\\\".format(
                _latex_label(sig),
                blob["return_ic"],
                blob["return_pval"],
                blob["abs_return_ic"],
                blob["abs_return_pval"],
            )
        )
    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        (
            f"\\caption{{Signal validity proxy: pooled Spearman information coefficients between "
            f"LLM signal coordinates and {lag}-day forward returns / absolute returns on the "
            f"2019--2023 stock-date panel. This is not a human annotation audit, but it checks "
            f"whether scores align with realized market quantities rather than pure noise.}}"
        ),
        r"\label{tab:signal_ic}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Signal} & \textbf{Return IC} & \textbf{$p$} & \textbf{$|$Return$|$ IC} & \textbf{$p$} \\",
        r"\midrule",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="trade_data_multi_signal_2019_2023.csv")
    parser.add_argument("--lag", type=int, default=5, help="Forward return horizon (trading days).")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: missing {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    results = {}
    for sig in SIGNAL_NAMES:
        if sig not in df.columns:
            print(f"  [skip] column missing: {sig}")
            continue
        blob = compute_ic(df, sig, lag=args.lag)
        results[sig] = blob
        print(
            f"  {sig:<28} return_IC={blob['return_ic']:.5f} "
            f"(p={blob['return_pval']:.3g})  "
            f"abs_return_IC={blob['abs_return_ic']:.5f} "
            f"(p={blob['abs_return_pval']:.3g})"
        )

    os.makedirs("backtest_results", exist_ok=True)
    path = "backtest_results/signal_ic_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"csv": args.csv, "lag": args.lag, "signals": results}, f, indent=2)
    print(f"\nSaved → {path}")
    tex_path = "paper/table_signal_ic.tex"
    write_latex_table(results, args.lag, tex_path)
    print(f"Saved → {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
