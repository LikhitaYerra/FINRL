#!/usr/bin/env python3
"""
FinBERT-SFP Baseline: Use FinBERT sentiment as direct portfolio ranking signal.

This addresses reviewer concern W3: the FinBERT comparison in Table 4 is only
for supervised ridge forecasting. The missing piece is using FinBERT as a
*direct factor portfolio signal* analogous to SFP.

Method:
  1. Load FinBERT scores from dense_text_panel_features.csv
  2. Merge with price data
  3. Rank stocks daily by finbert_sent
  4. Form top-10 long-only portfolio (same rule as SFP)
  5. Compare CR/Sharpe to four-factor SFP

This directly tests: dense neural text encoding vs. structured 4-axis SSAI
in the direct portfolio context (not just as ridge features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics_extended import compute_full_metrics

INITIAL_AMOUNT = 1_000_000


def run_finbert_portfolio(test_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Form daily top-k portfolio ranked by FinBERT sentiment."""
    dates = sorted(test_df["date"].unique())
    portfolio_values = [INITIAL_AMOUNT]
    holdings = []
    
    for i, date in enumerate(dates):
        day_data = test_df[test_df["date"] == date].copy()
        
        # Rank by FinBERT score (higher = more positive)
        # Handle NaN scores (treat as neutral = 0)
        day_data["finbert_sent"] = day_data["finbert_sent"].fillna(0.0)
        day_data = day_data.sort_values("finbert_sent", ascending=False)
        top_k = day_data.head(k)
        
        # Equal-weight top-k
        weights = np.ones(len(top_k)) / len(top_k)
        current_value = portfolio_values[-1]
        
        holdings.append({
            "date": date,
            "value": current_value,
            "tickers": ",".join(top_k["tic"].tolist()),
            "finbert_mean": top_k["finbert_sent"].mean(),
        })
        
        # Next-day return
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
    # Load price data
    test_prices = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    test_prices["date"] = pd.to_datetime(test_prices["date"]).dt.strftime("%Y-%m-%d")
    
    # Load FinBERT scores
    finbert_features = pd.read_csv("backtest_results/dense_text_panel_features.csv")
    finbert_features["date"] = pd.to_datetime(finbert_features["date"]).dt.strftime("%Y-%m-%d")
    finbert_features["tic"] = finbert_features["tic"].str.upper()
    
    # Merge
    test_df = test_prices.merge(
        finbert_features[["date", "tic", "finbert_sent"]],
        on=["date", "tic"],
        how="left"
    )
    
    print(f"Test data: {len(test_df)} rows")
    print(f"FinBERT coverage: {test_df['finbert_sent'].notna().sum()} / {len(test_df)} "
          f"({100*test_df['finbert_sent'].notna().mean():.1f}%)")
    print(f"FinBERT score range: [{test_df['finbert_sent'].min():.3f}, {test_df['finbert_sent'].max():.3f}]")
    
    print("\nRunning FinBERT-SFP portfolio (top-10)...")
    portfolio = run_finbert_portfolio(test_df, k=10)
    
    # Compute metrics
    portfolio["daily_return"] = portfolio["value"].pct_change().fillna(0.0)
    metrics = compute_full_metrics(portfolio["value"].values)
    
    print(f"\n{'='*60}")
    print("FinBERT-SFP Results (2019-2023)")
    print(f"{'='*60}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    # Save results
    portfolio.to_csv("backtest_results/finbert_sfp_portfolio.csv", index=False)
    print(f"\nSaved → backtest_results/finbert_sfp_portfolio.csv")
    
    return metrics


if __name__ == "__main__":
    import os
    os.makedirs("backtest_results", exist_ok=True)
    main()
