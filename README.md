# Semantic State Abstraction Interfaces (SSAI): Controlled Evaluation of LLM-Derived Signals in RL-Based Trading

[![Discord](https://dcbadge.limes.pink/api/server/ekrySuRBf4)](https://discord.gg/ekrySuRBf4)

| Resource | Link |
|----------|------|
| **Paper (HAL preprint)** | [hal.science/5613426](https://hal.science/view/index/docid/5613426) |
| **Paper PDF (this repo)** | [`paper/main.pdf`](paper/main.pdf) |
| **Public version (HAL)** | [`paper/main_hal.pdf`](paper/main_hal.pdf) |
| Blog | https://melwy.com/finrl_deepseek |
| arXiv (prior version) | https://arxiv.org/abs/2502.07393 |
| Hugging Face — data | [NASDAQ 2013–2023](https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/tree/main) |
| Hugging Face — agents | [Trading_agents](https://huggingface.co/benstaf/Trading_agents/tree/main) |
| FinRL upstream mirror | [AI4Finance FinRL_DeepSeek](https://github.com/AI4Finance-Foundation/FinRL_DeepSeek) |
| Contest | [FinRL Contest 2025 — Task 1](https://open-finance-lab.github.io/FinRL_Contest_2025/) |

Colab backtesting: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benstaf/FinRL_DeepSeek/blob/main/FinRL_DeepSeek_backtesting.ipynb)

---

## Research codebase (paper-oriented layout)

This repository extends the original release with a **unified evaluation harness**, **multi-seed reporting**, **SAM / ablation / regime** utilities, **strict out-of-sample** scripts, and **LaTeX table snippets** under `paper/`. For contribution descriptions and experiment semantics, read **`RESEARCH_GUIDE.md`**. For a reviewer-style checklist (data, checkpoints, commands), read **`REPRODUCIBILITY.md`**.

### Quick start — reproduce tables & figures

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
pip install -e spinningup_src/
```

Download **`train_data_multi_signal_2013_2018.csv`** and **`trade_data_multi_signal_2019_2023.csv`** to the repo root (HF link above). Obtain checkpoints from HF **Trading_agents** or train locally (`train_cppo_multi_signal.py` with MPI).

Then:

```bash
python multi_seed_eval.py --mode eval --seeds 5    # needs trained_models/seeds/agent_seed*.pth
python eval_harness.py                             # paper/tables + paper/figures
python signal_ic_report.py                         # pooled IC → backtest_results/signal_ic_report.json
python rolling_window_eval.py                      # per-year metrics → rolling_window_metrics.json
python run_pipeline.py --rigor                     # IC + rolling windows, then full pipeline stages
python run_pipeline.py                             # harness + SAM + seeds + lookahead + OOS + Uniswap (slow)
python run_pipeline.py --oos-score-news            # refresh LLM OOS CSV via OpenRouter, then OOS eval
python retrain_ablation_orchestrator.py --dry-run  # print MPI commands for train-time signal ablations
```

**Interactive dashboard:** see [`dashboard/README.md`](dashboard/README.md) — `./dashboard/start.sh` (FastAPI + Vite).

Environment variables for live news scoring: see **`.env.example`** (`OPENROUTER_API_KEY`).

### Headline comparisons (pick these first)

If you verify only **two** empirical claims about this fork, use:

| Claim | Command | Where it lands |
|--------|---------|----------------|
| **SAM vs concatenating LLM signals** | `python backtest_sam.py` | `backtest_results/sam_comparison.csv`, `paper/figures/fig_sam_comparison.png` |
| **Per-signal contribution (full trained agent)** | `python eval_harness.py` | `paper/table_ablation.tex`, `paper/figures/fig2_ablation.png` |

Optional standalone sweep (signals + simple baselines): `python ablation.py` → `backtest_results/ablation_results.csv`, `backtest_results/ablation_chart.png`. Semantics: **`RESEARCH_GUIDE.md`** (signal ablation section).

### Artifact map

| Output | Producer |
|--------|-----------|
| `paper/table_main.tex`, `paper/table_ablation.tex`, `paper/table_multiseed.tex`* | `eval_harness.py` |

\* `table_multiseed.tex` is emitted only if `backtest_results/multi_seed_results.json` exists (run `multi_seed_eval.py --mode eval` first).
| `paper/figures/fig1_equity_curves.png` (+ bands if present) | `eval_harness.py` |
| `backtest_results/multi_seed_results.json` | `multi_seed_eval.py --mode eval` (includes `bootstrap_ci_seeds` when ≥3 seeds) |
| `backtest_results/signal_ic_report.json` | `signal_ic_report.py` |
| `backtest_results/rolling_window_metrics.json` | `rolling_window_eval.py` |
| `trained_models/ablation_manifest.json` | `retrain_ablation_orchestrator.py` (after runs) |
| `backtest_results/oos_results.json`, `oos_portfolio.csv` | `oos_evaluation.py` |
| `oos_signals_2024_2025.csv` | `score_oos_news.py` |

---

## Results (original summary)

![Sample results](https://github.com/benstaf/FinRL_DeepSeek/blob/main/IMG_20250207_175434_001.jpg)

**Preliminary takeaway:** bull regimes favour plain PPO; drawdown-sensitive CPPO with DeepSeek-style signals can help in stressed regimes — validate on your own splits.

---

## Legacy installation & original training notes

Ubuntu server setup (MPI, conda, FinRL fork clone) is described in **`installation_script.sh`**.

### More details on dependencies

Run `installation_script.sh` on an Ubuntu server (**128 GB RAM** CPU instance recommended in the original workflow).

### Datasets and preprocessing (FNSPID path)

The baseline news corpus is **FNSPID** ([dataset](https://huggingface.co/datasets/Zihan1004/FNSPID), [paper](https://arxiv.org/abs/2402.06698)). LLM signal columns are produced with `sentiment_deepseek_deepinfra.py` / `risk_deepseek_deepinfra.py` and merged via `train_trade_data_deepseek_sentiment.py`, `train_trade_data_deepseek_risk.py`, or `train_trade_data.py` for non-LLM baselines.

### Training commands (classic variants)

- PPO: `mpirun --allow-run-as-root -np 8 python train_ppo.py …`
- CPPO: `train_cppo.py`
- PPO + LLM: `train_ppo_llm.py`
- CPPO + LLM risk: `train_cppo_llm_risk.py`
- **Multi-signal contest agent:** `train_cppo_multi_signal.py`

Environments: `env_stocktrading.py`, `env_stocktrading_llm*.py`, `env_stocktrading_llm_risk*.py`, **`env_stocktrading_multi_signal.py`**.

Monitor logs for **AverageEpRet**, **KL**, **ClipFrac**.

### Evaluation

Legacy evaluation can use **`FinRL_DeepSeek_backtesting.ipynb`**. This repo adds **`metrics_extended.py`** (Rachev, CVaR, Omega, Wilcoxon, etc.) and **`eval_harness.py`** for publication-ready exports.

---

## Citation

See **`CITATION.cff`** for GitHub/Zenodo-friendly metadata. BibTeX from the arXiv page is preferred for the PDF.
