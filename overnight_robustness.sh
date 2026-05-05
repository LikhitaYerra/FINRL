#!/usr/bin/env bash
# Overnight batch: more DP-PPO seeds + eval + robustness figures/tables.
# More seeds tighten seed-level statistics; they do not guarantee higher Sharpe.
#
# Usage (from repo root):
#   chmod +x overnight_robustness.sh
#   ./overnight_robustness.sh
# Or background:
#   mkdir -p logs && nohup ./overnight_robustness.sh >> logs/nohup_overnight.log 2>&1 &
#
# Env overrides:
#   SEEDS=24 EPOCHS=30 CPU=4 ./overnight_robustness.sh
#   OVERNIGHT_SKIP_TRAIN=1 ./overnight_robustness.sh   # only eval + robustness (needs checkpoints)

set -euo pipefail
export PYTHONUNBUFFERED=1
# macOS/Linux: prefer python3 (exit 127 if only `python` is missing)
PY="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PY" ]]; then
  echo "ERROR: no python3/python in PATH; set PYTHON=/path/to/python"
  exit 127
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-16}"
SEED_START="${SEED_START:-0}"
EPOCHS="${EPOCHS:-30}"
CPU="${CPU:-4}"
BOOTSTRAP="${BOOTSTRAP:-20000}"

mkdir -p logs
LOG="${ROOT}/logs/overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== overnight_robustness start $(date -u) ==="
echo "SEED_START=$SEED_START SEEDS=$SEEDS EPOCHS=$EPOCHS CPU=$CPU BOOTSTRAP=$BOOTSTRAP pwd=$ROOT"

if [[ ! -f train_data_multi_signal_2013_2018.csv ]]; then
  echo "WARN: train_data_multi_signal_2013_2018.csv missing — train step will fail until data exists."
fi
if [[ ! -f trade_data_multi_signal_2019_2023.csv ]]; then
  echo "WARN: trade_data_multi_signal_2019_2023.csv missing — eval will fail until data exists."
fi

if [[ "${OVERNIGHT_SKIP_TRAIN:-0}" != "1" ]]; then
  echo "--- multi_seed_eval TRAIN ---"
  "$PY" multi_seed_eval.py --mode train --seed-start "$SEED_START" --seeds "$SEEDS" --epochs "$EPOCHS" --cpu "$CPU"
else
  echo "--- skipping TRAIN (OVERNIGHT_SKIP_TRAIN=1) ---"
fi

echo "--- multi_seed_eval EVAL ---"
"$PY" multi_seed_eval.py --mode eval --seed-start "$SEED_START" --seeds "$SEEDS" --bootstrap-samples "$BOOTSTRAP"

echo "--- robustness_analysis ---"
"$PY" robustness_analysis.py

if [[ -f trained_models/agent_sac_llm_300_ep.pth ]]; then
  echo "--- backtest_sac (SAC vs DP-PPO, table + figure) ---"
  "$PY" backtest_sac.py || echo "WARN: backtest_sac failed (check trade CSV / checkpoints)"
fi

echo "=== overnight_robustness done $(date -u) log=$LOG ==="
