#!/usr/bin/env python3
"""
Run the main evaluation pipeline (stock RL + SAM + multi-seed + extensions).

Usage:
    python3 run_pipeline.py
    python3 run_pipeline.py --rigor               # signal IC + rolling-window diagnostics first
    python3 run_pipeline.py --skip harness       # skip slow eval_harness
    python3 run_pipeline.py --oos-score-news    # score_oos_news before OOS (needs OPENROUTER_API_KEY)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  FAILED ({label}) exit={r.returncode}", file=sys.stderr)
    return r.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="steps to skip: harness sam seeds sac_baseline lookahead oos uniswap",
    )
    parser.add_argument(
        "--oos-score-news",
        action="store_true",
        help="before OOS eval, run score_oos_news.py (requires .env / OPENROUTER_API_KEY unless CSV exists)",
    )
    parser.add_argument(
        "--rigor",
        action="store_true",
        help="run signal_ic_report.py + rolling_window_eval.py before other steps",
    )
    args = parser.parse_args()
    skip = set(args.skip)

    py = sys.executable
    steps = []

    if args.rigor:
        steps.append(([py, "signal_ic_report.py"], "signal_ic_report — pooled IC vs forward returns"))
        steps.append(([py, "rolling_window_eval.py"], "rolling_window_eval — calendar buckets on test window"))

    if "harness" not in skip:
        steps.append(([py, "eval_harness.py"], "eval_harness — strategies + tables + figures"))

    if "sam" not in skip:
        steps.append(([py, "backtest_sam.py"], "backtest_sam — CPPO vs SAM variants"))

    if "seeds" not in skip:
        steps.append(([py, "multi_seed_eval.py", "--mode", "eval", "--seeds", "16"],
                      "multi_seed_eval — aggregate checkpoints under trained_models/seeds/ (16 seeds)"))

    sac_ckpt = "trained_models/agent_sac_llm_300_ep.pth"
    if "sac_baseline" not in skip and os.path.isfile(sac_ckpt):
        steps.append(
            (
                [py, "backtest_sac.py"],
                "backtest_sac — SAC vs DP-PPO + fig_algorithm_comparison + table_algorithm_baseline.tex",
            )
        )

    if "lookahead" not in skip:
        steps.append(([py, "lookahead_bias.py", "--tests", "1,3"], "lookahead_bias — temporal IC split"))

    if "oos" not in skip:
        if args.oos_score_news:
            steps.append(([py, "score_oos_news.py"], "score_oos_news — refresh oos_signals_2024_2025.csv"))
        steps.append(([py, "oos_evaluation.py"], "oos_evaluation — 2024+ hold-out"))

    if "uniswap" not in skip:
        steps.append(([py, "train_uniswap_lp.py", "--mode", "eval"], "Uniswap LP eval"))

    failed = []
    for cmd, label in steps:
        if run(cmd, label) != 0:
            failed.append(label)

    print(f"\n{'═'*60}")
    if failed:
        print("  Pipeline finished with errors:")
        for f in failed:
            print(f"    • {f}")
        sys.exit(1)
    print("  Pipeline finished OK.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
