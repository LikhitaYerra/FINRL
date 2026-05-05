#!/usr/bin/env python3
"""
Train-time signal ablations (NeurIPS-style causal attribution).

Each variant runs ``train_cppo_multi_signal.py`` via MPI with different signal
overrides, saving to a distinct checkpoint suffix so policies are trained without
access to masked coordinates.

This is intentionally expensive (hours × variants). Use ``--dry-run`` to inspect commands.

Recommended workflow:
    # Inspect planned commands
    python retrain_ablation_orchestrator.py --dry-run --epochs 5 --cpu 4

    # Full overnight grid (example)
    python retrain_ablation_orchestrator.py --epochs 30 --cpu 4

Variants produced:
  - full                     → baselines multi-signal
  - scalar_sentiment        → only llm_sentiment varies
  - train_neutral_<signal>  → single axis neutralised during training

After training, evaluate each checkpoint with ``rolling_window_eval.py`` or a thin loop.

Outputs manifest:
    trained_models/ablation_manifest.json

Usage:
    python retrain_ablation_orchestrator.py [--epochs 30] [--cpu 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


VARIANTS: list[tuple[str, list[str]]] = [
    ("full", []),
    ("scalar_sentiment", ["--scalar_sentiment_only"]),
    ("neutral_sentiment", ["--neutral_llm_columns", "llm_sentiment"]),
    ("neutral_risk", ["--neutral_llm_columns", "llm_risk"]),
    ("neutral_confidence", ["--neutral_llm_columns", "llm_confidence"]),
    ("neutral_volatility_forecast", ["--neutral_llm_columns", "llm_volatility_forecast"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--train_csv", default="train_data_multi_signal_2013_2018.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print MPI commands without running.")
    args = parser.parse_args()
    dry = args.dry_run

    if not os.path.isfile(args.train_csv):
        print(f"ERROR: training CSV not found: {args.train_csv}")
        return 1

    py = sys.executable
    manifest = []

    for name, extra in VARIANTS:
        cmd = [
            "mpirun",
            "-np",
            str(args.cpu),
            py,
            "train_cppo_multi_signal.py",
            "--local_data",
            args.train_csv,
            "--epochs",
            str(args.epochs),
            "--cpu",
            str(args.cpu),
            "--seed",
            "0",
            "--exp_name",
            f"dppo_ablation_{name}",
            *extra,
        ]
        if name != "full":
            cmd.extend(["--save_suffix", "_" + name])
        manifest.append({"variant": name, "checkpoint_suffix": "(default)" if name == "full" else "_" + name, "cmd": cmd})
        print("\n" + "─" * 70)
        print(f"Variant: {name}")
        print(" ".join(cmd))
        if not dry:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"FAILED variant={name} exit={r.returncode}", file=sys.stderr)
                return r.returncode

    os.makedirs("trained_models", exist_ok=True)
    out = "trained_models/ablation_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"epochs": args.epochs, "variants": manifest}, f, indent=2)
    print(f"\nSaved manifest → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
