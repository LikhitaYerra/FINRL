#!/usr/bin/env python3
"""
Supervised forecasting baselines for semantic factor evaluation.

This script answers the reviewer request for a stronger non-RL baseline:
price/technical features are used to predict 5-day forward returns with ridge
regression, then the predictions rank stocks in the same daily top-10 portfolio
rule used by SFP. Variants compare price-only, price+LLM sentiment, optional lexical dense headlines
(VADER; train-fit TF--IDF/SVD; optional FinBERT), price+four-semantic features,
and a validation-selected semantic tilt.

Validation protocol:
  - Fit candidate ridge penalties on 2013--2017.
  - Select by validation Sharpe on 2018.
  - Refit selected model on 2013--2018.
  - Evaluate once on 2019--2023.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_extended import compute_full_metrics
from semantic_factor_portfolio import (
    INITIAL_AMOUNT,
    SIGNAL_COLS,
    _prepare,
    buy_hold_series,
    fit_semantic_weights,
    predict_scores as predict_semantic_scores,
    run_topk_portfolio,
)


PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "macd_scaled",
    "boll_pos",
    "rsi_30_scaled",
    "cci_30_scaled",
    "dx_30_scaled",
    "sma30_gap",
    "sma60_gap",
    "turbulence_scaled",
]

DENSE_PANEL_PATH = "backtest_results/dense_text_panel_features.csv"


def merge_dense_text_panel(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Left-merge optional dense/lexical headline features (see ``dense_text_baselines.py``)."""
    if not os.path.isfile(path):
        return df
    dense = pd.read_csv(path)
    dense["date"] = pd.to_datetime(dense["date"]).dt.strftime("%Y-%m-%d")
    dense["tic"] = dense["tic"].astype(str).str.strip().str.upper()
    meta = ["date", "tic"]
    feat_cols = [c for c in dense.columns if c not in meta]
    return df.merge(dense[meta + feat_cols], on=meta, how="left")


def dense_supervised_configs(train_merged: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Extra ridge configs when dense columns are present on the training panel."""
    train_view = train_merged.loc[train_merged["date"] < "2019-01-01"]
    extra: list[tuple[str, list[str]]] = []
    if "vader_compound" in train_merged.columns:
        extra.append(("Supervised price + VADER", PRICE_FEATURES + ["vader_compound"]))
    tfidf_cols = sorted(
        [c for c in train_merged.columns if c.startswith("tfidf_svd_")],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    if tfidf_cols:
        extra.append(("Supervised price + TF-IDF/SVD", PRICE_FEATURES + tfidf_cols))
    if "finbert_sent" in train_merged.columns:
        if train_view["finbert_sent"].notna().sum() >= max(500, len(train_view) // 50):
            extra.append(("Supervised price + FinBERT", PRICE_FEATURES + ["finbert_sent"]))
    return extra


def slug_metric_filename(strategy_name: str) -> str:
    s = strategy_name.lower().replace(" ", "_").replace("+", "plus")
    return s.replace("/", "_")


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged price/technical predictors available at each stock-day."""
    df = df.sort_values(["tic", "date"]).copy()
    grouped = df.groupby("tic", group_keys=False)
    df["ret_1d"] = grouped["close"].pct_change(1)
    df["ret_5d"] = grouped["close"].pct_change(5)
    df["ret_20d"] = grouped["close"].pct_change(20)
    df["vol_20d"] = grouped["close"].pct_change().rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    df["macd_scaled"] = df["macd"] / df["close"].replace(0.0, np.nan)
    denom = (df["boll_ub"] - df["boll_lb"]).replace(0.0, np.nan)
    df["boll_pos"] = (df["close"] - df["boll_lb"]) / denom
    df["rsi_30_scaled"] = (df["rsi_30"] - 50.0) / 50.0
    df["cci_30_scaled"] = df["cci_30"] / 200.0
    df["dx_30_scaled"] = df["dx_30"] / 100.0
    df["sma30_gap"] = df["close"] / df["close_30_sma"].replace(0.0, np.nan) - 1.0
    df["sma60_gap"] = df["close"] / df["close_60_sma"].replace(0.0, np.nan) - 1.0
    df["turbulence_scaled"] = df["turbulence"] / 100.0
    return df.replace([np.inf, -np.inf], np.nan)


def forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    close = df.pivot(index="date", columns="tic", values="close").sort_index()
    return close.pct_change(horizon).shift(-horizon).stack().rename("fwd_return")


def feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frames = []
    for col in cols:
        values = df.pivot(index="date", columns="tic", values=col).sort_index()
        frames.append(values.stack().rename(col))
    return pd.concat(frames, axis=1)


def fit_ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    ridge: float,
) -> tuple[np.ndarray, float, pd.Series, pd.Series]:
    data = x_train.join(y_train, how="inner").dropna()
    cols = list(x_train.columns)
    mu = data[cols].mean()
    sigma = data[cols].std().replace(0.0, 1.0).fillna(1.0)
    x = ((data[cols] - mu) / sigma).to_numpy(dtype=float)
    y = data["fwd_return"].to_numpy(dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(x_aug.shape[1]) * ridge
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return beta[1:], float(beta[0]), mu, sigma


def predict_scores(
    df: pd.DataFrame,
    x: pd.DataFrame,
    weights: np.ndarray,
    intercept: float,
    mu: pd.Series,
    sigma: pd.Series,
) -> pd.DataFrame:
    scores = df[["date", "tic", "close"]].drop_duplicates(["date", "tic"]).copy()
    scores = scores.set_index(["date", "tic"]).join(x, how="left")
    cols = list(x.columns)
    x_norm = ((scores[cols] - mu) / sigma).fillna(0.0).to_numpy(dtype=float)
    scores["score"] = intercept + x_norm @ weights
    return scores.reset_index()[["date", "tic", "close", "score"]]


def select_ridge(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    validation_start: str = "2018-01-01",
    grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> tuple[float, pd.DataFrame]:
    fit_df = train_df[train_df["date"] < validation_start]
    val_df = train_df[train_df["date"] >= validation_start]
    fit_x = feature_frame(fit_df, feature_cols)
    fit_y = forward_returns(fit_df)
    val_x = feature_frame(val_df, feature_cols)
    val_bh = buy_hold_series(val_df)

    rows = []
    for ridge in grid:
        weights, intercept, mu, sigma = fit_ridge(fit_x, fit_y, ridge=ridge)
        val_scores = predict_scores(val_df, val_x, weights, intercept, mu, sigma)
        curve = run_topk_portfolio(val_scores)
        metrics = compute_full_metrics(curve["portfolio_value"].values, val_bh["portfolio_value"].values, name="validation")
        rows.append({
            "ridge": ridge,
            "validation_cr": metrics["cumulative_return"],
            "validation_sharpe": metrics["sharpe_ratio"],
            "validation_mdd": metrics["max_drawdown_pct"],
        })
    result = pd.DataFrame(rows)
    best_idx = result["validation_sharpe"].idxmax()
    return float(result.loc[best_idx, "ridge"]), result


def zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("date")[col]
    return ((df[col] - grouped.transform("mean")) / grouped.transform("std").replace(0.0, np.nan)).fillna(0.0)


def semantic_tilt_scores(
    price_scores: pd.DataFrame,
    semantic_scores: pd.DataFrame,
    *,
    alpha: float,
    active_quantile: float = 0.67,
) -> pd.DataFrame:
    """
    Blend price forecasts with semantic scores only for high-conviction semantic names.

    The semantic term is active for the top tercile of absolute daily semantic
    z-scores. This avoids letting sparse, near-neutral semantic scores perturb
    every price forecast.
    """
    merged = price_scores.merge(
        semantic_scores[["date", "tic", "score"]].rename(columns={"score": "semantic_score"}),
        on=["date", "tic"],
        how="inner",
    )
    merged["price_z"] = zscore_by_date(merged, "score")
    merged["semantic_z"] = zscore_by_date(merged, "semantic_score")
    threshold = merged.groupby("date")["semantic_z"].transform(lambda s: s.abs().quantile(active_quantile))
    active = (merged["semantic_z"].abs() >= threshold).astype(float)
    out = merged[["date", "tic", "close"]].copy()
    out["score"] = merged["price_z"] + alpha * active * merged["semantic_z"]
    return out


def select_semantic_tilt_alpha(
    train_df: pd.DataFrame,
    *,
    price_ridge: float,
    validation_start: str = "2018-01-01",
    grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
) -> tuple[float, pd.DataFrame]:
    fit_df = train_df[train_df["date"] < validation_start]
    val_df = train_df[train_df["date"] >= validation_start]

    price_weights, price_intercept, price_mu, price_sigma = fit_ridge(
        feature_frame(fit_df, PRICE_FEATURES),
        forward_returns(fit_df),
        ridge=price_ridge,
    )
    price_val = predict_scores(
        val_df,
        feature_frame(val_df, PRICE_FEATURES),
        price_weights,
        price_intercept,
        price_mu,
        price_sigma,
    )
    sem_weights, sem_intercept, _ = fit_semantic_weights(fit_df, signal_cols=SIGNAL_COLS, strategy="semantic tilt validation")
    sem_val = predict_semantic_scores(val_df, sem_weights, sem_intercept, SIGNAL_COLS)
    val_bh = buy_hold_series(val_df)

    rows = []
    for alpha in grid:
        scores = semantic_tilt_scores(price_val, sem_val, alpha=alpha)
        curve = run_topk_portfolio(scores)
        metrics = compute_full_metrics(curve["portfolio_value"].values, val_bh["portfolio_value"].values, name="semantic tilt validation")
        rows.append({
            "alpha": alpha,
            "validation_cr": metrics["cumulative_return"],
            "validation_sharpe": metrics["sharpe_ratio"],
            "validation_mdd": metrics["max_drawdown_pct"],
        })
    result = pd.DataFrame(rows)
    best_idx = result["validation_sharpe"].idxmax()
    return float(result.loc[best_idx, "alpha"]), result


def write_latex_table(rows: list[dict], out_path: str) -> None:
    body = []
    for row in rows:
        body.append(
            "{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                row["Strategy"].replace("&", r"\&"),
                row["Tuned"],
                row["CR (%)"],
                row["Sharpe"],
                row["Sortino"],
                row["MDD (%)"],
                row["Calmar"],
            )
        )
    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Supervised forecasting baselines. Ridge models predict 5-day forward returns from price/technical features alone, price plus LLM sentiment or four-factor semantics, price plus lexical dense headlines (VADER; train-fit TF--IDF/SVD), optional FinBERT tone when available, or a semantic tilt. Ridge strength and the semantic tilt are selected on 2018 validation Sharpe, refit on 2013--2018, and evaluated once on 2019--2023 using the same daily top-10 portfolio rule as SFP.}",
        r"\label{tab:supervised_forecasting}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Tuned} & \textbf{CR (\%)} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{MDD (\%)} & \textbf{Calmar} \\",
        r"\midrule",
        "\n".join(body),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main() -> int:
    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    train = add_price_features(_prepare(pd.read_csv("train_data_multi_signal_2013_2018.csv")))
    test = add_price_features(_prepare(pd.read_csv("trade_data_multi_signal_2019_2023.csv")))
    train = merge_dense_text_panel(train, DENSE_PANEL_PATH)
    test = merge_dense_text_panel(test, DENSE_PANEL_PATH)
    dense_configs = dense_supervised_configs(train)
    test_bh = buy_hold_series(test)

    configs = [
        ("Supervised price-only", PRICE_FEATURES),
        ("Supervised price + sentiment", PRICE_FEATURES + ["llm_sentiment"]),
        *dense_configs,
        ("Supervised price + 4 semantic", PRICE_FEATURES + SIGNAL_COLS),
    ]

    metrics_rows = []
    all_curves = []
    validation_rows = []
    for name, cols in configs:
        ridge, val = select_ridge(train, cols)
        val.insert(0, "strategy", name)
        validation_rows.append(val)
        x_train = feature_frame(train, cols)
        y_train = forward_returns(train)
        weights, intercept, mu, sigma = fit_ridge(x_train, y_train, ridge=ridge)
        x_test = feature_frame(test, cols)
        scores = predict_scores(test, x_test, weights, intercept, mu, sigma)
        curve = run_topk_portfolio(scores)
        curve["strategy"] = name
        all_curves.append(curve)
        metrics = compute_full_metrics(curve["portfolio_value"].values, test_bh["portfolio_value"].values, name=name)
        metrics_rows.append({
            "Strategy": name,
            "Ridge": ridge,
            "Tuned": f"$\\lambda={ridge:.0e}$",
            "CR (%)": metrics["cumulative_return"],
            "Sharpe": metrics["sharpe_ratio"],
            "Sortino": metrics["sortino_ratio"],
            "MDD (%)": metrics["max_drawdown_pct"],
            "Calmar": metrics["calmar_ratio"],
        })
        coef = pd.DataFrame({"strategy": name, "feature": cols, "weight": weights})
        coef["intercept"] = intercept
        coef["ridge"] = ridge
        coef.to_csv(f"backtest_results/{slug_metric_filename(name)}_coefficients.csv", index=False)
        print(f"{name}: ridge={ridge:g} CR={metrics['cumulative_return']:.2f}% Sharpe={metrics['sharpe_ratio']:.3f}")

    price_ridge = float(pd.DataFrame(metrics_rows).loc[lambda d: d["Strategy"] == "Supervised price-only", "Ridge"].iloc[0])
    alpha, alpha_validation = select_semantic_tilt_alpha(train, price_ridge=price_ridge)
    alpha_validation.to_csv("backtest_results/supervised_semantic_tilt_validation.csv", index=False)

    price_weights, price_intercept, price_mu, price_sigma = fit_ridge(
        feature_frame(train, PRICE_FEATURES),
        forward_returns(train),
        ridge=price_ridge,
    )
    price_test = predict_scores(
        test,
        feature_frame(test, PRICE_FEATURES),
        price_weights,
        price_intercept,
        price_mu,
        price_sigma,
    )
    sem_weights, sem_intercept, _ = fit_semantic_weights(train, signal_cols=SIGNAL_COLS, strategy="semantic tilt")
    sem_test = predict_semantic_scores(test, sem_weights, sem_intercept, SIGNAL_COLS)
    tilt_scores = semantic_tilt_scores(price_test, sem_test, alpha=alpha)
    curve = run_topk_portfolio(tilt_scores)
    curve["strategy"] = "Supervised price + semantic tilt"
    all_curves.append(curve)
    metrics = compute_full_metrics(curve["portfolio_value"].values, test_bh["portfolio_value"].values, name="semantic tilt")
    metrics_rows.append({
        "Strategy": "Supervised price + semantic tilt",
        "Ridge": price_ridge,
        "Tuned": f"$\\lambda={price_ridge:.0e},\\alpha={alpha:g}$",
        "CR (%)": metrics["cumulative_return"],
        "Sharpe": metrics["sharpe_ratio"],
        "Sortino": metrics["sortino_ratio"],
        "MDD (%)": metrics["max_drawdown_pct"],
        "Calmar": metrics["calmar_ratio"],
    })
    print(
        f"Supervised price + semantic tilt: ridge={price_ridge:g} alpha={alpha:g} "
        f"CR={metrics['cumulative_return']:.2f}% Sharpe={metrics['sharpe_ratio']:.3f}"
    )

    m_bh = compute_full_metrics(test_bh["portfolio_value"].values, test_bh["portfolio_value"].values, name="Buy & Hold (EW)")
    metrics_rows.append({
        "Strategy": "Buy & Hold (EW)",
        "Ridge": 0.0,
        "Tuned": "--",
        "CR (%)": m_bh["cumulative_return"],
        "Sharpe": m_bh["sharpe_ratio"],
        "Sortino": m_bh["sortino_ratio"],
        "MDD (%)": m_bh["max_drawdown_pct"],
        "Calmar": m_bh["calmar_ratio"],
    })

    pd.DataFrame(metrics_rows).to_csv("backtest_results/supervised_forecasting_metrics.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv("backtest_results/supervised_forecasting_curves.csv", index=False)
    pd.concat(validation_rows, ignore_index=True).to_csv("backtest_results/supervised_forecasting_validation.csv", index=False)
    write_latex_table(metrics_rows, "paper/table_supervised_forecasting.tex")

    fig, ax = plt.subplots(figsize=(10, 4))
    for curve in all_curves:
        name = curve["strategy"].iloc[0]
        ax.plot(pd.to_datetime(curve["date"]), curve["portfolio_value"] / INITIAL_AMOUNT, lw=1.7, label=name)
    ax.plot(pd.to_datetime(test_bh["date"]), test_bh["portfolio_value"] / INITIAL_AMOUNT, lw=1.4, ls="--", color="#94A3B8", label="Buy & Hold (EW)")
    ax.set_title("Supervised Forecasting Baselines (2019--2023)", fontweight="bold")
    ax.set_ylabel("Normalised portfolio value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_supervised_forecasting.png", dpi=150)
    plt.close(fig)
    print("Saved supervised forecasting outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
