#!/usr/bin/env bash
#
# Optional bootstrap for Linux servers used with FinRL-DeepSeek.
# For day-to-day use, prefer the README path:
#   pip install -r requirements.txt && pip install -e spinningup_src/
#
set -euo pipefail

echo "=== FinRL-DeepSeek — conda + MPI (optional) ==="

if ! command -v conda &>/dev/null; then
  echo "Install Miniconda first, then re-run this script."
  echo "See: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

conda create --name finrl_ds python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finrl_ds

conda install -c conda-forge gcc swig mpi4py datasets huggingface_hub -y

pip install --upgrade pip
pip install -r requirements.txt
pip install -e spinningup_src/

echo ""
echo "Done. Activate with:  conda activate finrl_ds"
echo "Train CPPO multi-signal (example):"
echo "  OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \\"
echo "    mpirun -np 4 python train_cppo_multi_signal.py --epochs 30 --cpu 4"
