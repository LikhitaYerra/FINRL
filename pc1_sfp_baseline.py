#!/usr/bin/env python3
"""
PC1-SFP Baseline: Use first principal component as ranking signal.

This addresses reviewer concern W2: if four-factor SFP is essentially
one-dimensional (PC1 explains 82.1% variance), does the multi-axis
decomposition add value beyond what a single data-driven factor captures?

Method:
  1. Load the four semantic signals (sentiment, risk, confidence, vol)
  2. Compute PCA on training data (2013-2018)
  3. Project test data (2019-2023) onto PC1
  4. Rank stocks daily by PC1 score
  5. Form top-10 long-only portfolio (same rule as SFP)
  6. Compare CR/Sharpe to four-factor SFP

If PC1-SFP ≈ SFP: the multi-axis structure collapses to one factor
If SFP > PC1-SFP: the interpretable decomposition adds value
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from metrics_extended import compute_full_metrics

SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
INITIAL_AMOUNT = 1_000_000


def compute_pc1_scores(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Fit PCA on training data, project test data onto PC1."""
    # Extract signal matrices
    train_signals = train_df[SIGNAL_COLS].values - 3.0  # center around neutral
    test_signals = test_df[SIGNAL_COLS].values - 3.0
    
    # Fit PCA on training data
    pca = PCA(n_components=4)
    pca.fit(train_signals)
    
    # Project test data onto PC1
    test_pc1 = pca.transform(test_signals)[:, 0]
    
    print(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"PC1 loadings: sentiment={pca.components_[0,0]:.3f}, risk={pca.components_[0,1]:.3f}, "
          f"confidence={pca.components_[0,2]:.3f}, vol={pca.components_[0,3]:.3f}")
    
    return test_pc1, pca


def run_pc1_portfolio(test_df: pd.DataFrame, pc1_scores: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Form daily top-k portfolio ranked by PC1 score."""
    test_df = test_df.copy()
    test_df["pc1_score"] = pc1_scores
    
    dates = sorted(test_df["date"].unique())
    portfolio_values = [INITIAL_AMOUNT]
    holdings = []
    
    for i, date in enumerate(dates):
        day_data = test_df[test_df["date"] == date].copy()
        
        # Rank by PC1 score (higher = more positive sentiment-like)
        day_data = day_data.sort_values("pc1_score", ascending=False)
        top_k = day_data.head(k)
        
        # Equal-weight top-k
        weights = np.ones(len(top_k)) / len(top_k)
        current_value = portfolio_values[-1]
        
        holdings.append({
            "date": date,
            "value": current_value,
            "tickers": ",".join(top_k["tic"].tolist()),
            "pc1_mean": top_k["pc1_score"].mean(),
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
    # Load data
    train = pd.read_csv("train_data_multi_signal_2013_2018.csv")
    test = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    
    train["date"] = pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d")
    test["date"] = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
    
    # Fill missing signals with neutral
    for col in SIGNAL_COLS:
        train[col] = train[col].fillna(3.0)
        test[col] = test[col].fillna(3.0)
    
    print("Computing PC1 scores...")
    pc1_scores, pca = compute_pc1_scores(train, test)
    
    print("\nRunning PC1-SFP portfolio (top-10)...")
    portfolio = run_pc1_portfolio(test, pc1_scores, k=10)
    
    # Compute metrics
    portfolio["daily_return"] = portfolio["value"].pct_change().fillna(0.0)
    metrics = compute_full_metrics(portfolio["value"].values)
    
    print(f"\n{'='*60}")
    print("PC1-SFP Results (2019-2023)")
    print(f"{'='*60}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    # Save results
    portfolio.to_csv("backtest_results/pc1_sfp_portfolio.csv", index=False)
    print(f"\nSaved → backtest_results/pc1_sfp_portfolio.csv")
    
    return metrics


if __name__ == "__main__":
    import os
    os.makedirs("backtest_results", exist_ok=True)
    main()
