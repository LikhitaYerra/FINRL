#!/usr/bin/env python3
"""
Rolling calendar evaluation for a *fixed* trained checkpoint.

Splits the test CSV (`trade_data_multi_signal_2019_2023.csv`) into contiguous
calendar buckets (default: calendar years) and recomputes metrics vs equal-weight
buy-and-hold on each slice.

Outputs:
  backtest_results/rolling_window_metrics.json
  backtest_results/rolling_window_summary.csv

Usage:
    python rolling_window_eval.py
    python rolling_window_eval.py --model trained_models/agent_cppo_multi_signal_30_epochs.pth
    python rolling_window_eval.py --freq quarterly
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from eval_harness import (
    ACTION_SPACE,
    INITIAL_AMOUNT,
    STATE_SPACE,
    run_agent,
)
from metrics_extended import compute_full_metrics
from model_loader import load_cppo_model


def _prepare_trade(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    trade = df.copy()
    for col in ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    return trade, unique_dates


def _bh_series(trade: pd.DataFrame, n_dates: int) -> np.ndarray:
    avg = trade.reset_index().groupby("date")["close"].mean().sort_index().values
    avg = avg[:n_dates]
    return INITIAL_AMOUNT * avg / avg[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trade_csv",
        default="trade_data_multi_signal_2019_2023.csv",
        help="Test-period panel with LLM columns.",
    )
    parser.add_argument(
        "--model",
        default="trained_models/agent_cppo_multi_signal_30_epochs.pth",
        help="Single checkpoint to evaluate on each window.",
    )
    parser.add_argument(
        "--freq",
        choices=("year", "quarter"),
        default="year",
        help="Calendar aggregation granularity.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.trade_csv):
        print(f"ERROR: missing {args.trade_csv}")
        return 1
    if not os.path.isfile(args.model):
        print(f"ERROR: missing model {args.model}")
        return 1

    raw = pd.read_csv(args.trade_csv)
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    unique_dates = sorted(raw["date"].unique())

    dt_index = pd.to_datetime(unique_dates)
    if args.freq == "year":
        bucket_ids = dt_index.year.values
    else:
        bucket_ids = (dt_index.year.astype(str) + "Q" + dt_index.quarter.astype(str)).values

    trade_full = raw.copy()
    rows = []
    ac = load_cppo_model(args.model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    for bid in sorted(set(bucket_ids.tolist())):
        mask = bucket_ids == bid
        dates_sub = [d for d, m in zip(unique_dates, mask) if m]
        if len(dates_sub) < 20:
            continue
        sub_raw = trade_full[trade_full["date"].isin(dates_sub)].copy()
        trade_sub, u_sub = _prepare_trade(sub_raw)
        pvs = run_agent(ac, trade_sub, u_sub)
        bh = _bh_series(trade_sub, len(u_sub))
        m = compute_full_metrics(pvs, bh, name=f"DP-PPO ({bid})")
        rows.append(
            {
                "bucket": str(bid),
                "n_days": len(u_sub),
                "cumulative_return": m["cumulative_return"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sortino_ratio": m["sortino_ratio"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "calmar_ratio": m["calmar_ratio"],
                "wilcoxon_pval": m.get("wilcoxon_pval"),
            }
        )
        print(
            f"  {bid}: days={len(u_sub)}  CR={m['cumulative_return']:.2f}%  "
            f"Sharpe={m['sharpe_ratio']:.3f}  Wilcoxon p={m.get('wilcoxon_pval')}"
        )

    os.makedirs("backtest_results", exist_ok=True)
    out_json = "backtest_results/rolling_window_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "freq": args.freq, "windows": rows}, f, indent=2)
    print(f"\nSaved → {out_json}")

    pd.DataFrame(rows).to_csv("backtest_results/rolling_window_summary.csv", index=False)
    print("Saved → backtest_results/rolling_window_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
