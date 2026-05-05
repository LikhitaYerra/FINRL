# NeurIPS 2026 Submission Guide — DEADLINE: May 5th

## Files in this folder
| File | Purpose |
|------|---------|
| `main.tex` | Full paper source (NeurIPS 2024 template) |
| `references.bib` | Bibliography |
| `table_main.tex`, `table_ablation.tex`, `table_multiseed.tex` | Numeric tables (regenerate via `eval_harness.py`; `\input{}` from `main.tex`) |
| `neurips_2024.sty` | Official NeurIPS 2024 LaTeX style |
| `figures/` | Equity / ablation PNGs from `eval_harness.py` (`fig1_equity_curves.png`, etc.) |
| `main.pdf` | Compiled PDF — verify before submission |

---

## Step 1 — Polish on Overleaf (10 min)

1. Go to https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
2. Click **"Open as Template"**
3. Upload `main.tex`, `references.bib`, `table_*.tex`, `neurips_2024.sty`, and any generated PNGs under `figures/`
4. Overleaf has required packages (`booktabs`, etc.).
5. Run `python eval_harness.py` from the **repository root** (with data + checkpoint) to refresh `paper/table_main.tex`, `paper/table_ablation.tex`, and `paper/figures/fig1_equity_curves.png`. `main.tex` includes the equity figure only if `figures/fig1_equity_curves.png` exists (`\IfFileExists`).
6. **Local pdflatex:** if Helvetica metrics are missing, `main.tex` falls back to CM sans so the PDF still builds; on Overleaf you typically get full NeurIPS fonts without changes.
7. Keep the author block anonymized for double-blind review; update it only for a non-anonymous preprint or camera-ready.
8. Download final PDF

---

## Step 2 — OpenReview Profile (5 min)

1. Go to https://openreview.net/signup
2. Register with your **AIvancity email** (e.g., likhita.yerra@aivancity.ai)
3. Fill in institution: **AIvancity**
4. Verify email
5. Complete profile: add Google Scholar/DBLP links if available

---

## Step 3 — NeurIPS 2026 Submission on OpenReview (15 min)

1. Go to https://openreview.net/group?id=NeurIPS.cc/2026/Conference
2. Click **"Submit"** → **"NeurIPS 2026 Conference Submission"**
3. Fill in:
   - **Title**: LLM Semantic Factor Interfaces for Reinforcement Learning Portfolio Trading: Multi-Signal News Decomposition under Stochastic RL
   - **Abstract**: (copy from paper)
   - **Keywords**: reinforcement learning, large language models, semantic state abstraction, portfolio management, financial NLP
   - **TL;DR**: We study four-axis LLM semantic factorization as an auditable state interface for direct portfolios and RL; direct semantic portfolios show the clearest gains, while RL benefits are optimizer-dependent and underpowered across current seeds.
   - **Authors**: Add authors in OpenReview metadata, but keep the uploaded PDF anonymized if the venue requires double-blind review.
4. Upload the PDF
5. Submit before **May 5th 23:59 AoE**

---

## Step 4 — HAL Submission with arXiv Cross-Post

1. **Create HAL account**: https://hal.science → "Create account"
2. **Submit**:
   - Go to https://hal.science → "Submit a document"
   - Type: Conference paper (preprint)
   - Upload PDF
   - Fill metadata: title, abstract, keywords, author(s), institution: AIvancity
   - Domain: Computer Science > Artificial Intelligence
3. **Enable arXiv cross-post**:
   - During submission, check "Cross-post to arXiv"
   - HAL will request your arXiv account email for endorsement
   - If no arXiv account, create one at https://arxiv.org/register
   - Category: cs.LG (Machine Learning) + q-fin.TR (Trading and Market Microstructure)
4. Submit and note the HAL ID (format: hal-XXXXXXX)
5. arXiv cross-post appears within ~24 hours as arXiv:XXXX.XXXXX

---

## Key Results (for abstract/submission form)

- Four-factor SFP: 307.2% cumulative return, Sharpe 1.067.
- Sentiment-only SFP: 291.9% cumulative return, Sharpe 1.050.
- Equal-weight buy-and-hold: 243.6% cumulative return, Sharpe 1.032.
- SCW: 314.1% cumulative return via validation-selected conviction weighting.
- SAC with the same semantic state: 242.7% cumulative return, Sharpe 1.128.
- Eight-seed DP-PPO table: full semantic inputs improve mean return and Sharpe over neutral-masked evaluation, but paired seed tests remain non-significant ($p{\approx}0.25$--$0.31$).

Initial capital: $1,000,000 | Test period: 2019-01-02 to 2023-12-29 | 1,258 trading days

---

## ⚠️ Deadline reminder

Confirm the official venue deadline on OpenReview and update this guide if dates change.
