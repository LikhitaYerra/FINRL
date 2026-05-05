# Reproducibility checklist

Use this for **paper reviewers**, **your future self**, and **benchmark reproduction**.

## 1. Environment

- **Python:** 3.10 or 3.11 (3.11 works with the pinned stack below).
- **Why `numpy<2`:** Several optional stacks (OpenCV ↔ Gym, TensorBoard ↔ TensorFlow paths) still assume NumPy 1.x. The root `requirements.txt` pins accordingly.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e spinningup_src/
```

Spinning Up dependencies are declared in `spinningup_src/setup.py` (PyTorch + `mpi4py` + `gymnasium`, no TensorFlow 1.x).

## 2. Data files (repo root)

Large CSVs are **gitignored** by design. Download to the project root:

| File | Role |
|------|------|
| `train_data_multi_signal_2013_2018.csv` | Training |
| `trade_data_multi_signal_2019_2023.csv` | In-sample backtest / harness |

Source: [Hugging Face dataset](https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/tree/main) (see main README).

## 3. Checkpoints

Pre-trained weights are **gitignored** (`*.pth`). Obtain via:

- [Hugging Face — Trading_agents](https://huggingface.co/benstaf/Trading_agents/tree/main), **or**
- Train locally, e.g. `train_cppo_multi_signal.py` with MPI (see `installation_script.sh` for conda/MPI hints).

Multi-seed aggregation expects:

`trained_models/seeds/agent_seed{0..4}.pth`

Generate with:

```bash
python multi_seed_eval.py --mode train --seeds 5 --epochs 30
```

(or train seeds manually; see project logs).

Then:

```bash
python multi_seed_eval.py --mode eval --seeds 5
```

Outputs: `backtest_results/multi_seed_results.json`, `multi_seed_portfolio.csv`.

## 4. Paper tables and figures

Single entrypoint:

```bash
python eval_harness.py
```

Produces LaTeX snippets under `paper/` (`table_main.tex`, `table_ablation.tex`, and **`table_multiseed.tex`** when `multi_seed_results.json` exists), plots under `paper/figures/` (when not skipped). Figure 1 picks up **multi-seed bands** automatically if `multi_seed_portfolio.csv` exists.

Full sweep (optional):

```bash
python run_pipeline.py
python run_pipeline.py --oos-score-news   # refresh OOS LLM CSV before OOS eval; needs API key
```

## 5. Out-of-sample (2024+)

1. Optional LLM refresh: copy `.env.example` → `.env`, set `OPENROUTER_API_KEY`, then `python score_oos_news.py`.
2. Run `python oos_evaluation.py` (`--signals auto` uses `oos_signals_2024_2025.csv` if present).

See `oos_evaluation.py` docstring for merge semantics (point-in-time vs snapshot broadcast).

## 6. NeurIPS-style empirical diagnostics (optional)

- **`python signal_ic_report.py`** — pooled Spearman IC between each LLM axis and forward returns → `backtest_results/signal_ic_report.json`.
- **`python rolling_window_eval.py`** — same checkpoint evaluated on calendar buckets of the test CSV → `backtest_results/rolling_window_metrics.json` (+ CSV summary).
- **`python run_pipeline.py --rigor`** — runs the two scripts above before the heavier harness steps.
- **`python retrain_ablation_orchestrator.py --dry-run`** — prints MPI training commands for **train-time** masking ablations (full grid is compute-heavy). Manifest: `trained_models/ablation_manifest.json`.
- **`multi_seed_eval.py --mode eval`** — writes bootstrap 95% intervals on the **mean Sharpe / mean cumulative return across seeds** into `multi_seed_results.json` under `bootstrap_ci_seeds` when at least three seed checkpoints exist.

Training-time controls on **`train_cppo_multi_signal.py`**:

- `--scalar_sentiment_only` — only sentiment varies; other LLM axes fixed at neutral during training.
- `--neutral_llm_columns llm_risk,...` — listed columns forced to 3.0 for the full train window.
- `--save_suffix _custom` — distinct checkpoint filename (`agent_cppo_multi_signal_{epochs}e_custom.pth`).

## 7. What to archive for a submission supplement

- `requirements.txt` (+ exact `pip freeze` in supplemental text).
- `multi_seed_results.json`, `full_results_table.csv`, `oos_results.json`.
- Versioned copies of `paper/figures/*` used in the PDF.
- Seed integers and `spinningup_src` commit hash / FinRL fork revision.

For deeper methodology notes, see **`RESEARCH_GUIDE.md`**.
