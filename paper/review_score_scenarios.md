# Review-Score Scenario Probe

**Purpose:** This file is for stress-testing LLM reviewer reactions only.
It is **not a manuscript**, **not a PDF source**, and **not for submission**.
The "predicted" seed numbers below are hypothetical scenarios derived from the current 5-seed pattern; they are not experimental results and must not be copied into `main.tex`, `main.pdf`, OpenReview, arXiv, HAL, or any artifact as factual results.

## Current Paper Snapshot

Title: **LLM Semantic Factor Interfaces for Reinforcement Learning Portfolio Trading: Multi-Signal News Decomposition under Stochastic RL**

Core framing:

> Sequential decision systems increasingly observe sparse, unstructured text, but it remains unclear whether large language models (LLMs) should be used as dense black-box embeddings, scalar sentiment, or structured state abstractions. The paper studies four-axis LLM semantic factorization as an auditable state interface for direct semantic portfolios and RL agents.

Main contribution claims:

1. **Semantic state abstraction:** four interpretable LLM axes for ticker-day decision states: sentiment, risk, confidence, and volatility forecast.
2. **Isolation tests:** SFP, SRF, SCW, and supervised semantic tilt separate semantic factor usefulness from high-variance RL training.
3. **Representation-versus-optimization diagnostics:** the same semantic state is evaluated in DP-PPO and SAC with masking, sub-period, transaction-cost, and multi-seed diagnostics.

Current real results:

- Four-factor SFP: 307.2% cumulative return, Sharpe 1.067.
- Sentiment-only SFP: 291.9% cumulative return, Sharpe 1.050.
- Equal-weight buy-and-hold: 243.6% cumulative return, Sharpe 1.032.
- SCW: 314.1% cumulative return.
- Supervised price-only ridge: 126.4% cumulative return, Sharpe 0.653.
- Supervised price + semantic tilt: 133.9% cumulative return, Sharpe 0.674.
- SAC with identical semantic state: 242.7% cumulative return, Sharpe 1.128.
- Current real DP-PPO seed table: 5 seeds only.

Current real 5-seed DP-PPO aggregate:

| Variant | Cumulative Return | Sharpe |
|---|---:|---:|
| DP-PPO full semantic inputs | 192.6% ± 31.8% | 0.912 ± 0.100 |
| DP-PPO neutral-masked eval | 183.3% ± 28.9% | 0.876 ± 0.085 |

Current real 5-seed bootstrap intervals:

| Metric | Full | Neutral eval |
|---|---:|---:|
| Mean CR 95% CI | [167.0%, 221.8%] | [158.5%, 208.2%] |
| Mean Sharpe 95% CI | [0.828, 1.002] | [0.804, 0.948] |

Current honest interpretation:

> Direct semantic portfolios provide the strongest positive evidence. RL gains are directional but underpowered; the five-seed DP-PPO result improves over neutral-masked evaluation on average, but the seed count is too small for strong statistical claims. SAC shows that optimizer choice matters under the same semantic state.

## Scenario A: Current Submitted Version

**Use case:** Ask an LLM reviewer to score the actual current paper.

Assumed paper state:

- 5 DP-PPO seeds.
- Direct SFP/SRF/SCW results included.
- Supervised semantic tilt included.
- SAC baseline included.
- No FinBERT/FinGPT/Transformer baseline.
- No prompt/model robustness study.
- No human annotation validation.

Likely reviewer response:

- Strengths: clear framing, strong decomposition philosophy, honest limitations, good reproducibility.
- Weaknesses: limited ML novelty, weak statistical power, missing stronger text baselines, limited LLM validation.
- Plausible score: **5/10 to 6/10**, depending on reviewer fit.

## Scenario B: Hypothetical 10-Seed Conservative Projection

**Important:** This is a projection, not a result.

Hypothesis:

- Seeds 5--9 behave similarly to the current seed distribution.
- Mean values stay close to current 5-seed aggregate.
- Confidence intervals tighten only modestly.
- Paired full-vs-neutral tests remain non-significant or only marginal.

Projected wording for review testing only:

> With 10 seeds, DP-PPO full semantic inputs remain directionally stronger than neutral-masked evaluation, with mean cumulative return around 190--198% and Sharpe around 0.90--0.93, versus neutral-masked return around 181--187% and Sharpe around 0.86--0.89. The paired seed tests remain underpowered, but the larger sample reduces concern that the 5-seed pattern is purely accidental.

Expected review effect:

- Fixes the "only five seeds" optics partially.
- Does not fully solve statistical significance.
- Plausible score movement: **5/10 -> 5.5--6/10**.

## Scenario C: Hypothetical 10-Seed Favorable Projection

**Important:** This is a projection, not a result.

Hypothesis:

- Additional seeds preserve positive full-vs-neutral deltas.
- Mean full semantic return rises modestly or remains stable.
- Neutral-masked evaluation remains lower.

Projected wording for review testing only:

> With 10 seeds, full semantic DP-PPO obtains mean cumulative return around 200--210% and Sharpe around 0.94--0.98, compared with neutral-masked evaluation around 180--188% and Sharpe around 0.86--0.89. The full-vs-neutral delta is consistent across most seeds, making the semantic-state effect more credible even if daily-return tests remain noisy.

Expected review effect:

- Stronger response to "weak statistical rigor."
- Still vulnerable to "missing strong baselines."
- Plausible score movement: **5/10 -> 6/10**.

## Scenario D: Hypothetical 15-Seed Strong Projection

**Important:** This is a projection, not a result.

Hypothesis:

- Seeds 5--14 mostly show positive full-vs-neutral deltas.
- Mean delta in cumulative return is roughly 20--30 percentage points.
- Sharpe delta is roughly 0.07--0.10.
- Paired seed-level test becomes at least marginal, possibly significant.

Projected wording for review testing only:

> Across 15 DP-PPO seeds, full semantic inputs produce a stable improvement over neutral-masked evaluation, with mean cumulative return around 205--215% versus 180--188% and Sharpe around 0.95--1.00 versus 0.86--0.89. Paired seed-level tests indicate that the effect is no longer only a single-checkpoint artifact, although the RL policy still trails the direct semantic portfolio and remains optimizer-dependent.

Expected review effect:

- Much better response to "five seeds is too small."
- Makes the RL section look like a real robustness study rather than only a diagnostic.
- Still does not address FinBERT/FinGPT/Transformer or LLM robustness.
- Plausible score movement: **5/10 -> 6--6.5/10**.

## Scenario E: Hypothetical Stronger-Baseline Add-On

**Important:** This is a design scenario, not a completed experiment.

Add one of:

- FinBERT sentiment baseline aggregated to ticker-day and used in the same SFP pipeline.
- TF-IDF/Ridge headline baseline using article text to predict 5-day forward returns.
- Simple LSTM/Transformer price-only forecaster plus semantic tilt.

Projected reviewer effect:

> The baseline suite becomes more credible because the paper no longer compares only against passive, rule-based, ridge, PPO, and SAC baselines. Even if the proposed method does not dominate every baseline, the comparison makes the scope clearer and reduces the "missing standard NLP baseline" objection.

Expected score movement if results are honest and reasonably competitive:

- Current + one text baseline: **5.5--6.5/10**.
- Current + 10--15 seeds + one text baseline: **6--7/10** if the new evidence is not negative.

## Recommended Prompt for LLM Review Testing

Paste the current paper or a summary plus one scenario, then ask:

> Review this NeurIPS-style paper under the following evidence condition. Treat all numbers in the "Scenario" section as hypothetical for scoring analysis only. What score would you likely assign, and what objections remain?

Suggested scenario labels:

- "Current 5-seed version"
- "Hypothetical conservative 10-seed update"
- "Hypothetical favorable 10-seed update"
- "Hypothetical strong 15-seed update"
- "Hypothetical 15-seed + FinBERT baseline update"

## Do Not Do This

Do **not**:

- Copy scenario numbers into `main.tex`.
- Generate a PDF manuscript with scenario numbers.
- Paste scenario numbers into OpenReview as real results.
- Rename this file as an appendix or supplement.

Only replace paper numbers after the corresponding checkpoints, logs, tables, and regenerated artifacts exist.
