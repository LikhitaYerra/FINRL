#!/usr/bin/env python3
"""
Softmax-SFP Baseline (equal-weight auditable combination).

Reviewer Q1 asks: if PC1 loadings are ~equal across axes, can a fully
auditable equal-weight combination recover PC1 performance?

Method:
  1. Standardise the four SSAI axes using training-period mean/std
  2. Rank stocks daily by the simple mean of standardised axes
  3. Form top-10 long-only portfolio (same rule as SFP / PC1-SFP)
  4. This is fully auditable: the composite = (sent + risk + conf + vol) / 4
     in normalised units — no post-hoc compression required

If Softmax-SFP ≈ PC1-SFP: PC1 adds no interpretability cost.
If Softmax-SFP < PC1-SFP: PCA rotation matters beyond equal weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metrics_extended import compute_full_metrics

SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
INITIAL_AMOUNT = 1_000_000


def run_softmax_portfolio(test_df: pd.DataFrame, scores: np.ndarray, k: int = 10) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["composite_score"] = scores

    dates = sorted(test_df["date"].unique())
    portfolio_values = [INITIAL_AMOUNT]
    holdings = []

    for i, date in enumerate(dates):
        day_data = test_df[test_df["date"] == date].copy()
        day_data = day_data.sort_values("composite_score", ascending=False)
        top_k = day_data.head(k)
        weights = np.ones(len(top_k)) / len(top_k)
        current_value = portfolio_values[-1]

        holdings.append({
            "date": date,
            "value": current_value,
            "tickers": ",".join(top_k["tic"].tolist()),
            "score_mean": top_k["composite_score"].mean(),
        })

        if i < len(dates) - 1:
            next_date = dates[i + 1]
            returns = []
            for tic, w in zip(top_k["tic"], weights):
                today_close = day_data[day_data["tic"] == tic]["close"].values[0]
                next_day = test_df[(test_df["date"] == next_date) & (test_df["tic"] == tic)]
                if len(next_day) > 0:
                    next_close = next_day["close"].values[0]
                    ret = (next_close - today_close) / today_close
                    returns.append(w * ret)
            portfolio_return = sum(returns) if returns else 0.0
            portfolio_values.append(current_value * (1 + portfolio_return))

    return pd.DataFrame(holdings)


def main():
    train = pd.read_csv("train_data_multi_signal_2013_2018.csv")
    test  = pd.read_csv("trade_data_multi_signal_2019_2023.csv")

    train["date"] = pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d")
    test["date"]  = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")

    for col in SIGNAL_COLS:
        train[col] = train[col].fillna(3.0)
        test[col]  = test[col].fillna(3.0)

    # Fit scaler on training period, apply to test
    scaler = StandardScaler()
    scaler.fit(train[SIGNAL_COLS].values)
    test_scaled = scaler.transform(test[SIGNAL_COLS].values)

    # Equal-weight composite score
    composite_scores = test_scaled.mean(axis=1)

    print("Running Softmax-SFP (equal-weight auditable) portfolio (top-10)…")
    portfolio = run_softmax_portfolio(test, composite_scores, k=10)

    portfolio["daily_return"] = portfolio["value"].pct_change().fillna(0.0)
    metrics = compute_full_metrics(portfolio["value"].values)

    print(f"\n{'='*60}")
    print("Softmax-SFP Results (2019–2023)")
    print(f"{'='*60}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.1f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%")

    portfolio.to_csv("backtest_results/softmax_sfp_portfolio.csv", index=False)
    print(f"\nSaved → backtest_results/softmax_sfp_portfolio.csv")
    return metrics


if __name__ == "__main__":
    import os
    os.makedirs("backtest_results", exist_ok=True)
    main()
