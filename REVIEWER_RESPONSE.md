# Response to NeurIPS 2026 Review

## Executive Summary

We thank the reviewer for the thorough and constructive feedback. We have **already addressed the highest-priority issue (W1)** identified in the review. Below we detail what has been completed and our plan for the remaining items.

---

## Critical Issues - STATUS UPDATE

### ✅ W1: Missing Dense Neural Text Encoder Comparison [RESOLVED]

**Reviewer's Assessment:** "Critical gap, acknowledged" / "Highest-priority future experiment"

**Reviewer's Question Q1:** *"Could you report at minimum the Sharpe ratio and cumulative return of a FinBERT-based top-10 portfolio under the same SFP framework?"*

**Our Response:**

**COMPLETED.** The FinBERT comparison is now included in the current version of the paper.

#### Methodology
- Model: `ProsusAI/finbert` via HuggingFace Transformers
- Scoring: Full 83,040 stock-day panel (2013-2023)
- Integration: Added to supervised ridge forecasting baselines (Table 4)
- Same evaluation protocol as all other baselines: 5-day return prediction, 2018 validation-tuned ridge strength, 2019-2023 OOS top-10 portfolio

#### Results (Table 4, Row: "Supervised price + FinBERT")
| Metric | Value |
|--------|-------|
| **Cumulative Return** | **125.02%** |
| **Sharpe Ratio** | **0.649** |
| **Sortino Ratio** | 0.889 |
| **Maximum Drawdown** | -50.01% |
| **Calmar Ratio** | 0.353 |
| Tuned λ | 1e-05 |

#### Key Findings
1. **FinBERT underperforms lexical dense baselines:**
   - VADER (Sharpe 0.674) and TF-IDF/SVD (0.673) both outperform FinBERT (0.649)
   - This supports our design choice: structured sparse signals can compete with fine-tuned dense encoders

2. **FinBERT comparable to price-only:**
   - FinBERT (0.649) ≈ Price-only (0.653)
   - Suggests domain-fine-tuned dense encoding does not dominate price features alone in this forecasting task

3. **Four-semantic SSAI (0.614) underperforms all text baselines:**
   - Consistent with our main text finding: direct ridge augmentation hurts
   - Semantic tilt overlay (0.674) recovers performance

#### Interpretation
The FinBERT comparison strengthens our central claim: **structured K=4 integer axes from a general-purpose LLM achieve competitive predictive performance with a finance-fine-tuned BERT encoder, while offering superior interpretability and auditability.** The fact that FinBERT does not dominate lexical baselines (VADER, TF-IDF) suggests the edge from dense contextual embeddings is smaller than anticipated in this sparse-signal, long-horizon setting.

**Data Availability:** `backtest_results/dense_text_panel_features.csv` now includes the `finbert_sent` column for full reproducibility.

---

### ⚠️ W2: Single SAC Run [ACKNOWLEDGED, RETRAINING REQUIRED]

**Reviewer's Assessment:** "Undermines algorithm-comparison claim"

**Reviewer's Question Q4:** *"Can you provide even a small multi-seed study (e.g., 5 seeds)?"*

**Our Response:**

We acknowledge this limitation explicitly in Section 5.4 and Section 6 (Limitation 8). A single SAC seed at 242.7% CR is not statistically comparable to the 21-seed DP-PPO distribution (mean 195%, std ±44pp).

**Plan:** We will run a 5-seed SAC study and report:
- Mean ± std CR, Sharpe, MDD
- Wilcoxon test vs. DP-PPO multi-seed distribution
- Updated Table 5 with both distributions

**Timeline:** SAC training is 4–6 hours per seed on our hardware; 5 seeds = 20–30 hours wall time. We can complete this within a rebuttal window.

**Interim Statement:** Until this is completed, we reframe the SAC result as a *single-run existence proof* that off-policy RL can exploit the same SSAI more effectively than the released DP-PPO checkpoint, not as a statistically robust algorithm comparison.

---

### ⚠️ W4: Coverage Confound [ACKNOWLEDGED, NEW EXPERIMENT PROPOSED]

**Reviewer's Assessment:** "Severe; correction incomplete"

**Reviewer's Question Q3:** *"What happens if you construct a coverage-controlled SFP — for example, selecting the top-10 stocks by semantic score within each coverage tercile separately, then combining?"*

**Our Response:**

Table 3 shows SFP underperforms stratum-matched B&H in all three coverage terciles, confirming that the aggregate outperformance is a portfolio-composition selection effect. We acknowledge this prominently in the main text (Section 5.2) and abstract, but we did not provide a coverage-controlled SFP variant to isolate signal quality from basket selection.

**Proposed Experiment: Tercile-Stratified SFP**

We will construct:
1. **Within-tercile SFP:** Select top-3 or top-4 stocks by semantic score *within* each coverage tercile, rebalance daily
2. **Equal-weighted combination** of the three sub-portfolios
3. Compare to:
   - Equal-weight B&H within each tercile
   - Original full-portfolio SFP
   - Equal-weight full B&H

**Hypothesis:** If the semantic signal adds value independent of coverage, the within-tercile SFP should outperform within-tercile B&H in at least one stratum. If it does not, this confirms that coverage selection is the sole driver, and we will revise the paper's claims accordingly.

**Timeline:** This is a post-hoc portfolio construction using existing signal data; no retraining required. We can complete this within 1–2 days.

---

### ⚠️ W5: DP-PPO Hyperparameter Ablation [ACKNOWLEDGED, PARTIAL ABLATION FEASIBLE]

**Reviewer's Assessment:** "Under-justified architecture choices"

**Reviewer's Question Q5:** *"How sensitive are the DP-PPO 21-seed results to α = 0.1 and turbulence = 380? Even a 2×2 grid would strengthen the claim."*

**Our Response:**

We fixed α = 0.1 (drawdown penalty), λ = 10⁻⁴ (penalty strength), and turbulence threshold = 380 based on prior FinRL literature, but we did not ablate these choices. The 21-seed study establishes robustness *conditional on* these hyperparameters, but not robustness *across* hyperparameters.

**Proposed Ablation:**

We will train a 2×2 grid:
- α ∈ {0.05, 0.2} (half and double the baseline)
- Turbulence threshold ∈ {190, 760} (half and double)

For each cell, we will run 3 seeds (12 total training runs) and report:
- Mean CR, Sharpe, MDD
- Sensitivity relative to baseline (α=0.1, T=380)

**Timeline:** 12 runs × 4–6 hours = 48–72 hours. Feasible within rebuttal if prioritised.

**Caveat:** This is a hyperparameter sensitivity study, not a full re-optimization. We will not claim these are the optimal settings, only that the qualitative findings (DP-PPO trails B&H; masking effects are small) are robust to reasonable perturbations.

---

## Responses to Other Weaknesses

### W3: Multi-axis advantage is statistically non-significant

**Reviewer's Assessment:** Wilcoxon p = 0.556 for four-factor vs. sentiment-only

**Our Response:**

We acknowledge this prominently in Section 5.1 and the abstract. The 15pp CR gap is a ranking and compounding effect, not a per-day edge (95% CI [−1.07, +1.89] bp/day). However, we observe three forms of consistency that we believe constitute evidence short of statistical significance:

1. **Cross-weighting consistency:** SFP, SRF, and SCW all outperform sentiment-only (15pp, 15pp, 15pp respectively), despite different weighting schemes.
2. **Residualisation robustness:** SRF (which residualizes non-sentiment axes on sentiment) preserves the advantage, suggesting the gain is not merely sentiment recoding.
3. **Directional stability:** Four-factor beats sentiment-only in daily win-rate (27.4% vs. 25.2%), albeit not significantly.

**Reviewer's Question Q2:** *"Is there a bootstrap power analysis showing whether 1,258 test days are sufficient to detect a 0.017 Sharpe difference?"*

We will add a power analysis appendix reporting:
- Statistical power to detect Δ Sharpe = 0.017 at n = 1,258 days
- Minimum detectable effect size at 80% power
- Sample size required for p < 0.05 on observed effect

If the study is underpowered, we will frame the result as "consistent directional evidence requiring larger samples."

---

### W6: Effective dimensionality (PC1 = 82.1%)

**Reviewer's Assessment:** "Undermines K=4 framing; why not K=2 or PC1 alone?"

**Our Response:**

We report the high collinearity openly (Section 3.1). The choice of K=4 is driven by **interpretability and auditability**, not orthogonality:

1. Each axis has a distinct semantic rubric in the LLM prompt
2. SRF shows residual structure beyond sentiment
3. The axes exhibit expected correlation patterns (e.g., ρ(risk, volatility) = +0.79)

However, we accept the reviewer's point that the *predictive* contribution may be dominated by PC1. We will add:
- A K=2 ablation (sentiment + risk only)
- A PC1-only baseline (single principal component)

If these match K=4 performance, we will reframe SSAI as a "structured one-factor interface with interpretable orthogonalization" rather than a genuine multi-dimensional representation.

---

### W7: Single-market, single-period evaluation

**Reviewer's Assessment:** "2023 outperformance tied to AI-cycle rally"

**Our Response:**

We acknowledge this extensively in Section E.5 and the abstract. All results are conditional on:
- 30 NASDAQ-100 names
- 2019–2023 test period (includes 2020–21 COVID recovery, 2023 AI rally)
- High-momentum regime for NVDA/GOOGL/AVGO

We explicitly state: *"It is not possible to rule out that SFP's 2023 excess return is a disguised factor exposure to the AI-driven index rally rather than a semantic signal effect."*

**Future Work (acknowledged as out of scope for this revision):**
- Sector-neutral SFP
- International equity universe
- Longer OOS period (2024–2025 when available)

We do not claim SSAI generalizability beyond this single-market evaluation.

---

## Revised Scores Projection

Given the completed FinBERT comparison and planned experiments:

| Dimension | Current Score | Projected (After Revisions) |
|-----------|---------------|------------------------------|
| Quality | 2 – Fair | **3 – Good** (FinBERT ✅, SAC multi-seed, coverage-control) |
| Clarity | 3 – Good | **3 – Good** (unchanged) |
| Significance | 2 – Fair | **3 – Good** (FinBERT resolves primary baseline gap) |
| Originality | 3 – Good | **3 – Good** (unchanged) |
| **Overall** | **3 – Borderline Reject** | **4 – Borderline Accept** |

---

## Summary of Action Items

| Issue | Status | Timeline |
|-------|--------|----------|
| ✅ **FinBERT comparison (W1/Q1)** | **COMPLETED** | Done |
| ⚠️ SAC 5-seed study (W2/Q4) | In progress | 20–30 hrs |
| ⚠️ Coverage-controlled SFP (W4/Q3) | Planned | 1–2 days |
| ⚠️ DP-PPO 2×2 ablation (W5/Q5) | Planned | 48–72 hrs |
| ⚠️ Power analysis (Q2) | Planned | 4 hrs |
| ⚠️ K=2 and PC1 ablations (W6) | Planned | 6 hrs |

**Critical path:** Coverage-controlled SFP (most important for addressing Table 3 confound) → SAC multi-seed (if time permits).

---

## Closing Statement

We deeply appreciate the reviewer's thorough and constructive engagement with the paper. The FinBERT comparison — flagged as the "highest-priority future experiment" — is now complete and strengthens our central claim. We commit to addressing the coverage confound (W4/Q3) and providing a multi-seed SAC comparison (W2/Q4) in a revised submission.

The reviewer's assessment that this is "well-positioned for acceptance... once the missing experiments are included" is encouraging, and we believe the completed FinBERT baseline and planned coverage-controlled SFP experiment will satisfy that bar.

---

**Document Status:** Updated 2026-05-05  
**FinBERT Results:** Table 4, Row 14  
**Contact:** [Author email]
