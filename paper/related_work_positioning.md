# Related Work Positioning

## How our paper extends / differs from each reference

---

### FinRL-DeepSeek (Benhenda 2025, arXiv:2502.07393)
**Our direct baseline.**

| Aspect | FinRL-DeepSeek | Our Work |
|--------|---------------|----------|
| LLM signals | 1 (sentiment) | **4** (sentiment, risk, confidence, vol forecast) |
| Signal integration | Concatenation | **Cross-attention (SAM)** |
| Risk constraint | None | **CVaR penalty + drawdown** |
| Evaluation period | 2019-2021 | 2019-2023 + **2024-2025 OOS** |
| Look-ahead test | Not reported | **Explicit test — no bias found** |
| Ablation | None | **Per-signal quantitative ablation** |
| Statistical test | None | **5-seed Wilcoxon test** |
| Regime analysis | None | **HMM 3-state regime detection** |

We extend FinRL-DeepSeek along every evaluative dimension while introducing
the Signal Attention Module as a novel architectural contribution.

---

### Look-Ahead-Bench (Benhenda 2026, arXiv:2601.13770)
**Motivates our look-ahead bias analysis section.**

Look-Ahead-Bench establishes that LLMs with post-event training cutoffs can
exhibit inflated predictive performance on historical financial data.

**Our test results (Section 4.5 in the paper):**
- Pre-cutoff IC (2013-2017): Sentiment=0.013, Risk=-0.012, Confidence=0.009
- Post-cutoff IC (2017-2023): Sentiment=0.006, Risk=-0.010, Confidence=0.007
- IC ratio (post/pre): max = 0.81 — **DECREASING over time**
- Verdict: No look-ahead bias. IC values (0.007-0.010) are in the range of
  genuine news alpha, well below the 0.05+ typical of look-ahead contamination.

We follow the Look-Ahead-Bench methodology for temporal IC splitting and
cross-model validation.

---

### FutureX (Zeng et al. 2025, arXiv:2508.11987) + Chandak et al. 2026 (arXiv:2512.25070)
**Motivates our out-of-sample live evaluation.**

FutureX and the Chandak et al. scaling work establish the importance of
live, point-in-time evaluation of LLM predictions. We apply this philosophy
to financial RL: train on 2013-2018, evaluate in-sample on 2019-2023,
then run a STRICT out-of-sample test on 2024-2025.

**OOS Results (Section 5, Table 3):**
- CPPO agent: CR=103.6%, Annual=43.0%, Sharpe=1.217
- Buy & Hold: CR=43.5%,  Annual=19.9%, Sharpe=0.945
- **Alpha: +60.1pp CR, +0.272 Sharpe — with neutral LLM signals**
- OOS Sharpe (1.217) > in-sample Sharpe (1.070) — no overfitting detected

This is the strongest result in the paper: the agent generalizes to
18 months of unseen data it could not have been trained on.

---

### VCBench (Chen et al. 2025, arXiv:2509.14448) + YCBench (Benhenda 2026, arXiv:2604.02378)
**Methodological context for LLM financial signal evaluation.**

VCBench and YCBench establish best practices for benchmarking LLMs in
financial prediction tasks. We follow their recommendations:
1. Point-in-time evaluation (no future news in scoring context)
2. Multi-model comparison (safe vs risk model IC comparison in Section 4.5)
3. Coverage reporting (Section 4.1: 14-20% non-neutral signal coverage)
4. Temporal stability analysis (IC timeseries in Figure 7)

---

### Uniswap LP with RL (Xu & Brini 2025, arXiv:2501.07508)
**Motivates DeFi as a second environment.**

Xu & Brini demonstrate RL viability for DeFi liquidity provisioning — 
a structurally different financial environment from equities. Future work
will apply our LLM-signal-augmented CPPO to the Uniswap v3 LP problem,
testing whether the SAM architecture generalizes across:
  - Asset type: equities vs. DeFi tokens
  - Action space: discrete positions vs. continuous LP range selection  
  - Signal type: company news vs. on-chain analytics

---

## Summary Table

| Paper | Role in our work |
|-------|----------------|
| FinRL-DeepSeek | Baseline we extend |
| Look-Ahead-Bench | Bias test methodology |
| FutureX + Chandak | Live evaluation design |
| VCBench + YCBench | Signal evaluation framework |
| Uniswap LP | Future work direction |
