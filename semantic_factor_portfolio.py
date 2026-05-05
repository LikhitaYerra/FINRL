#!/usr/bin/env python3
"""
Semantic Factor Portfolio (SFP) baseline.

This is a lightweight, non-RL baseline that treats the four LLM coordinates as
explicit semantic factors. It learns linear factor weights on the training panel
to predict forward returns, then forms a daily long-only top-k portfolio on the
2019--2023 test panel.

Why this matters:
  - Provides a direct test of the semantic decomposition outside RL.
  - Adds a single-sentiment comparator without retraining RL policies.
  - Adds Semantic Residual Factorization (SRF): non-sentiment axes are
    residualized against sentiment on the training panel, testing whether
    risk/confidence/volatility contain information beyond scalar sentiment.
  - Adds Regime-Conditional SFP (RC-SFP): separate semantic factor weights are
    learned for low- and high-volatility regimes, testing whether the meaning
    of the same LLM axes changes across market states.
  - Adds Semantic Conviction Weighting (SCW): validation-selected softmax
    concentration over the top semantic names, testing whether score magnitude
    should affect allocation size rather than only rank.
  - Helps separate "useful semantic factors" from "RL optimizer effects".

Outputs:
  backtest_results/semantic_factor_portfolio.csv
  backtest_results/semantic_factor_weights.csv
  paper/table_semantic_coverage_sensitivity.tex
  paper/table_semantic_factor_portfolio.tex
  paper/figures/fig_semantic_factor_weights.png
  paper/figures/fig_semantic_factor_portfolio.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_extended import compute_full_metrics


SIGNAL_COLS = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
INITIAL_AMOUNT = 1_000_000


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in SIGNAL_COLS:
        df[col] = df.get(col, 3.0).fillna(3.0)
    return df.sort_values(["date", "tic"])


def _feature_frame(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    close = df.pivot(index="date", columns="tic", values="close").sort_index()
    frames = []
    for col in signal_cols:
        sig = df.pivot(index="date", columns="tic", values=col).sort_index().reindex(close.index)
        frames.append((sig - 3.0).stack().rename(col))
    return pd.concat(frames, axis=1)


def build_residualized_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Semantic Residual Factorization (SRF).

    Keep sentiment as the scalar anchor. For risk/confidence/volatility, fit
    x_j = a_j + b_j * sentiment on the training panel and use residuals as
    additional semantic axes. This explicitly asks whether the extra axes carry
    information not linearly explained by scalar sentiment.
    """
    base_cols = SIGNAL_COLS
    train_x = _feature_frame(train_df, base_cols)
    target_x = _feature_frame(target_df, base_cols)
    sent_train = train_x["llm_sentiment"].to_numpy(dtype=float)
    sent_target = target_x["llm_sentiment"].to_numpy(dtype=float)

    train_res = pd.DataFrame(index=train_x.index)
    target_res = pd.DataFrame(index=target_x.index)
    train_res["sentiment"] = train_x["llm_sentiment"]
    target_res["sentiment"] = target_x["llm_sentiment"]

    params = []
    x_aug = np.column_stack([np.ones(len(sent_train)), sent_train])
    for col in ["llm_risk", "llm_confidence", "llm_volatility_forecast"]:
        y = train_x[col].to_numpy(dtype=float)
        coef = np.linalg.lstsq(x_aug, y, rcond=None)[0]
        name = col.replace("llm_", "") + "_resid"
        train_res[name] = y - x_aug @ coef
        target_res[name] = target_x[col].to_numpy(dtype=float) - (
            np.column_stack([np.ones(len(sent_target)), sent_target]) @ coef
        )
        params.append({
            "axis": name,
            "intercept_vs_sentiment": float(coef[0]),
            "slope_vs_sentiment": float(coef[1]),
        })
    return train_res, target_res, pd.DataFrame(params)


def _forward_returns(train_df: pd.DataFrame, horizon: int) -> pd.Series:
    close = train_df.pivot(index="date", columns="tic", values="close").sort_index()
    fwd = close.pct_change(horizon).shift(-horizon)
    y = fwd.stack().rename("fwd_return")
    return y


def fit_weights_from_features(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    ridge: float,
    strategy: str,
    horizon: int,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Fit ridge weights from arbitrary semantic feature columns."""
    data = x.join(y, how="inner").dropna()
    cols = list(x.columns)

    x_mat = data[cols].to_numpy(dtype=float)
    y_vec = data["fwd_return"].to_numpy(dtype=float)
    x_aug = np.column_stack([np.ones(len(x_mat)), x_mat])
    reg = np.eye(x_aug.shape[1]) * ridge
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_vec)
    intercept = float(beta[0])
    weights = beta[1:]

    weights_df = pd.DataFrame({
        "strategy": strategy,
        "signal": cols,
        "weight": weights,
        "horizon_days": horizon,
        "ridge": ridge,
        "n_train_pairs": len(data),
    })
    return weights, intercept, weights_df


def fit_semantic_weights(
    train_df: pd.DataFrame,
    *,
    horizon: int = 5,
    ridge: float = 1e-3,
    signal_cols: list[str] | None = None,
    strategy: str = "SFP",
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Fit a ridge regression from signal deviations to forward returns."""
    signal_cols = signal_cols or SIGNAL_COLS
    x = _feature_frame(train_df, signal_cols)
    y = _forward_returns(train_df, horizon)
    return fit_weights_from_features(x, y, ridge=ridge, strategy=strategy, horizon=horizon)


def predict_scores_from_features(test_df: pd.DataFrame, x: pd.DataFrame, weights: np.ndarray, intercept: float) -> pd.DataFrame:
    scores = test_df[["date", "tic", "close"]].drop_duplicates(["date", "tic"]).copy()
    scores = scores.set_index(["date", "tic"]).join(x, how="left").reset_index()
    feature_cols = list(x.columns)
    scores[feature_cols] = scores[feature_cols].fillna(0.0)
    x_mat = scores[feature_cols].to_numpy(dtype=float)
    scores["score"] = intercept + x_mat @ weights
    return scores[["date", "tic", "close", "score"]]


def predict_scores(test_df: pd.DataFrame, weights: np.ndarray, intercept: float, signal_cols: list[str]) -> pd.DataFrame:
    scores = test_df[["date", "tic", "close"] + signal_cols].copy()
    x = (scores[signal_cols] - 3.0).to_numpy(dtype=float)
    scores["score"] = intercept + x @ weights
    return scores


def volatility_regimes(
    df: pd.DataFrame,
    *,
    threshold: float | None = None,
    window: int = 20,
) -> tuple[pd.Series, float, pd.Series]:
    """Classify dates into low/high volatility regimes using past equal-weight returns."""
    close = df.pivot(index="date", columns="tic", values="close").sort_index()
    ew_ret = close.pct_change().mean(axis=1)
    realized_vol = ew_ret.rolling(window, min_periods=5).std().shift(1)
    realized_vol = realized_vol.fillna(realized_vol.expanding(min_periods=1).mean()).fillna(0.0)
    if threshold is None:
        threshold = float(realized_vol.median())
    regimes = pd.Series(
        np.where(realized_vol >= threshold, "high_vol", "low_vol"),
        index=realized_vol.index,
        name="regime",
    )
    return regimes, threshold, realized_vol


def fit_regime_conditional_sfp(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    horizon: int = 5,
    ridge: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit regime-specific semantic weights and score the test panel.

    The volatility threshold is estimated on the training period. Test regimes
    use only lagged test-window equal-weight returns, so the scorer does not use
    future labels or forward returns at evaluation time.
    """
    train_x = _feature_frame(train_df, SIGNAL_COLS)
    test_x = _feature_frame(test_df, SIGNAL_COLS)
    y = _forward_returns(train_df, horizon)
    train_regime, threshold, train_vol = volatility_regimes(train_df)
    test_regime, _, test_vol = volatility_regimes(test_df, threshold=threshold)

    weights_by_regime: dict[str, tuple[np.ndarray, float]] = {}
    weight_frames = []
    for regime in ["low_vol", "high_vol"]:
        dates = train_x.index.get_level_values("date")
        mask = dates.map(train_regime).to_numpy() == regime
        if mask.sum() < len(SIGNAL_COLS) * 10:
            mask = np.ones(len(train_x), dtype=bool)
        weights, intercept, weights_df = fit_weights_from_features(
            train_x.loc[mask],
            y,
            ridge=ridge,
            strategy=f"RC-SFP ({regime})",
            horizon=horizon,
        )
        weights_df["intercept"] = intercept
        weights_df["vol_threshold"] = threshold
        weights_df["n_regime_dates"] = int((train_regime == regime).sum())
        weights_by_regime[regime] = (weights, intercept)
        weight_frames.append(weights_df)

    scores = test_df[["date", "tic", "close"]].drop_duplicates(["date", "tic"]).copy()
    scores = scores.set_index(["date", "tic"]).join(test_x, how="left")
    scores[SIGNAL_COLS] = scores[SIGNAL_COLS].fillna(0.0)
    date_index = scores.index.get_level_values("date")
    scores["regime"] = date_index.map(test_regime).fillna("low_vol")
    scores["score"] = 0.0

    for regime, (weights, intercept) in weights_by_regime.items():
        mask = scores["regime"] == regime
        x_mat = scores.loc[mask, SIGNAL_COLS].to_numpy(dtype=float)
        scores.loc[mask, "score"] = intercept + x_mat @ weights

    regime_trace = pd.DataFrame({
        "date": test_regime.index,
        "regime": test_regime.values,
        "lagged_ew_vol": test_vol.values,
        "train_vol_threshold": threshold,
    })
    return scores.reset_index()[["date", "tic", "close", "score", "regime"]], pd.concat(weight_frames, ignore_index=True), regime_trace


def run_topk_portfolio(
    scores: pd.DataFrame,
    *,
    top_k: int = 10,
    cost_pct: float = 0.001,
) -> pd.DataFrame:
    close = scores.pivot(index="date", columns="tic", values="close").sort_index()
    score = scores.pivot(index="date", columns="tic", values="score").reindex(close.index).fillna(0.0)
    rets = close.pct_change().fillna(0.0)

    weights_prev = pd.Series(0.0, index=close.columns)
    values = []
    turnovers = []
    value = INITIAL_AMOUNT
    for date in close.index:
        daily_ret = float((weights_prev * rets.loc[date]).sum())
        value *= (1.0 + daily_ret)

        ranked = score.loc[date].sort_values(ascending=False)
        chosen = ranked.head(top_k).index
        weights_new = pd.Series(0.0, index=close.columns)
        weights_new.loc[chosen] = 1.0 / top_k
        turnover = float((weights_new - weights_prev).abs().sum())
        value *= max(0.0, 1.0 - cost_pct * turnover)

        values.append({"date": date, "portfolio_value": value, "turnover": turnover})
        turnovers.append(turnover)
        weights_prev = weights_new
    return pd.DataFrame(values)


def run_conviction_weighted_portfolio(
    scores: pd.DataFrame,
    *,
    top_k: int = 10,
    temperature: float = 500.0,
    cost_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Allocate inside the top-k basket according to semantic score conviction.

    Equal-weight SFP only uses scores for ranking. SCW additionally uses their
    relative magnitudes: higher semantic scores receive larger long-only weights
    through a softmax over the selected basket.
    """
    close = scores.pivot(index="date", columns="tic", values="close").sort_index()
    score = scores.pivot(index="date", columns="tic", values="score").reindex(close.index).fillna(0.0)
    rets = close.pct_change().fillna(0.0)

    weights_prev = pd.Series(0.0, index=close.columns)
    values = []
    value = INITIAL_AMOUNT
    for date in close.index:
        daily_ret = float((weights_prev * rets.loc[date]).sum())
        value *= (1.0 + daily_ret)

        chosen = score.loc[date].sort_values(ascending=False).head(top_k)
        logits = np.clip((chosen - chosen.mean()) * temperature, -50.0, 50.0)
        raw = np.exp(logits)
        weights_new = pd.Series(0.0, index=close.columns)
        weights_new.loc[chosen.index] = raw / raw.sum()
        turnover = float((weights_new - weights_prev).abs().sum())
        value *= max(0.0, 1.0 - cost_pct * turnover)

        values.append({"date": date, "portfolio_value": value, "turnover": turnover})
        weights_prev = weights_new
    return pd.DataFrame(values)


def select_conviction_temperature(
    train_df: pd.DataFrame,
    *,
    validation_start: str = "2018-01-01",
    grid: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0),
) -> tuple[float, pd.DataFrame]:
    """Select SCW temperature on the final training year by validation Sharpe."""
    fit_df = train_df[train_df["date"] < validation_start]
    val_df = train_df[train_df["date"] >= validation_start]
    if fit_df.empty or val_df.empty:
        return 100.0, pd.DataFrame()

    weights, intercept, _ = fit_semantic_weights(fit_df, signal_cols=SIGNAL_COLS, strategy="SCW validation")
    val_scores = predict_scores(val_df, weights, intercept, SIGNAL_COLS)
    val_bh = buy_hold_series(val_df)
    rows = []
    for temperature in grid:
        curve = run_conviction_weighted_portfolio(val_scores, temperature=temperature)
        m = compute_full_metrics(curve["portfolio_value"].values, val_bh["portfolio_value"].values, name="SCW validation")
        rows.append({
            "temperature": temperature,
            "validation_cr": m["cumulative_return"],
            "validation_sharpe": m["sharpe_ratio"],
            "validation_mdd": m["max_drawdown_pct"],
        })
    results = pd.DataFrame(rows)
    best_idx = results["validation_sharpe"].idxmax()
    return float(results.loc[best_idx, "temperature"]), results


def buy_hold_series(test_df: pd.DataFrame) -> pd.DataFrame:
    avg = test_df.groupby("date")["close"].mean().sort_index()
    value = INITIAL_AMOUNT * (avg / avg.iloc[0])
    return pd.DataFrame({"date": value.index, "portfolio_value": value.values})


def ticker_signal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Return ticker-level fraction of stock-days with any non-neutral signal."""
    rows = []
    for tic, group in df.groupby("tic"):
        signal_frame = group[SIGNAL_COLS]
        non_neutral = (signal_frame.ne(3.0)).any(axis=1)
        rows.append({
            "tic": tic,
            "coverage_pct": float(non_neutral.mean() * 100.0),
            "n_days": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)


def coverage_sensitivity_rows(test_df: pd.DataFrame, scores: pd.DataFrame, weights: np.ndarray, intercept: float) -> list[dict]:
    """
    Compare SFP on high-coverage and low-coverage ticker subsets.

    This tests whether the direct semantic portfolio result is concentrated in
    names with more frequent non-neutral LLM observations.
    """
    coverage = ticker_signal_coverage(test_df)
    median_coverage = float(coverage["coverage_pct"].median())
    coverage["coverage_group"] = np.where(coverage["coverage_pct"] >= median_coverage, "High coverage", "Low coverage")
    coverage.to_csv("backtest_results/semantic_ticker_coverage.csv", index=False)

    rows = []
    for group_name in ["High coverage", "Low coverage"]:
        tickers = coverage.loc[coverage["coverage_group"] == group_name, "tic"].tolist()
        subset_test = test_df[test_df["tic"].isin(tickers)]
        subset_scores = scores[scores["tic"].isin(tickers)]
        curve = run_topk_portfolio(subset_scores, top_k=min(10, len(tickers)))
        bh = buy_hold_series(subset_test)
        m = compute_full_metrics(curve["portfolio_value"].values, bh["portfolio_value"].values, name=f"SFP {group_name}")
        rows.append({
            "Coverage group": group_name,
            "Tickers": len(tickers),
            "Coverage (%)": float(coverage.loc[coverage["coverage_group"] == group_name, "coverage_pct"].mean()),
            "CR (%)": m["cumulative_return"],
            "Sharpe": m["sharpe_ratio"],
            "MDD (%)": m["max_drawdown_pct"],
        })
    return rows


def write_latex_table(rows: list[dict], out_path: str) -> None:
    body = []
    for row in rows:
        body.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                row["Strategy"].replace("&", r"\&"),
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
        r"\caption{Semantic Factor Portfolio variants. Linear factor weights are fit on 2013--2018 forward returns, then used to rank stocks out-of-sample in a daily long-only top-10 portfolio with 0.1\% transaction costs. SRF residualizes non-sentiment axes against sentiment. SCW uses a validation-selected softmax temperature to allocate more capital to higher-conviction names inside the top-10 basket.}",
        r"\label{tab:semantic_factor_portfolio}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{CR (\%)} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{MDD (\%)} & \textbf{Calmar} \\",
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


def write_coverage_table(rows: list[dict], out_path: str) -> None:
    body = []
    for row in rows:
        body.append(
            "{} & {} & {:.1f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                row["Coverage group"],
                row["Tickers"],
                row["Coverage (%)"],
                row["CR (%)"],
                row["Sharpe"],
                row["MDD (%)"],
            )
        )
    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Coverage sensitivity for the four-factor SFP baseline. Tickers are split by median out-of-sample non-neutral signal coverage, and each subset is evaluated against its own equal-weight buy-and-hold sleeve.}",
        r"\label{tab:semantic_coverage_sensitivity}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Tickers} & \textbf{Coverage (\%)} & \textbf{CR (\%)} & \textbf{Sharpe} & \textbf{MDD (\%)} \\",
        r"\midrule",
        "\n".join(body),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def plot_factor_weights(weights_df: pd.DataFrame, out_path: str) -> None:
    sfp = weights_df[weights_df["strategy"] == "SFP (4 factors)"].copy()
    sfp["signal"] = sfp["signal"].str.replace("llm_", "", regex=False).str.replace("_", " ", regex=False)
    colors = np.where(sfp["weight"] >= 0.0, "#2563EB", "#DC2626")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(sfp["signal"], sfp["weight"], color=colors)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_title("Learned Semantic Factor Weights", fontweight="bold")
    ax.set_ylabel("Ridge weight")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

    train = _prepare(pd.read_csv("train_data_multi_signal_2013_2018.csv"))
    test = _prepare(pd.read_csv("trade_data_multi_signal_2019_2023.csv"))
    bh = buy_hold_series(test)

    configs = [
        ("SFP (4 factors)", SIGNAL_COLS),
        ("SFP (sentiment only)", ["llm_sentiment"]),
    ]

    metrics_rows = []
    all_curves = []
    all_weights = []
    for name, cols in configs:
        weights, intercept, weights_df = fit_semantic_weights(train, signal_cols=cols, strategy=name)
        weights_df["intercept"] = intercept
        all_weights.append(weights_df)

        scores = predict_scores(test, weights, intercept, cols)
        if name == "SFP (4 factors)":
            sfp_scores = scores.copy()
            sfp_weights = weights.copy()
            sfp_intercept = intercept
        curve = run_topk_portfolio(scores)
        curve["strategy"] = name
        all_curves.append(curve)
        m = compute_full_metrics(curve["portfolio_value"].values, bh["portfolio_value"].values, name=name)
        metrics_rows.append({
            "Strategy": name,
            "CR (%)": m["cumulative_return"],
            "Sharpe": m["sharpe_ratio"],
            "Sortino": m["sortino_ratio"],
            "MDD (%)": m["max_drawdown_pct"],
            "Calmar": m["calmar_ratio"],
        })
        print(f"{name}: CR={m['cumulative_return']:.2f}% Sharpe={m['sharpe_ratio']:.3f}")

    # Semantic Residual Factorization: extra axes orthogonalized against sentiment.
    train_res, test_res, residual_params = build_residualized_features(train, test)
    weights, intercept, weights_df = fit_weights_from_features(
        train_res,
        _forward_returns(train, horizon=5),
        ridge=1e-3,
        strategy="SRF (sentiment + residual axes)",
        horizon=5,
    )
    weights_df["intercept"] = intercept
    all_weights.append(weights_df)
    residual_params.insert(0, "strategy", "SRF residualization")
    residual_params.to_csv("backtest_results/semantic_residualization_params.csv", index=False)

    scores = predict_scores_from_features(test, test_res, weights, intercept)
    curve = run_topk_portfolio(scores)
    curve["strategy"] = "SRF (sentiment + residual axes)"
    all_curves.append(curve)
    m = compute_full_metrics(curve["portfolio_value"].values, bh["portfolio_value"].values, name="SRF")
    metrics_rows.append({
        "Strategy": "SRF (sentiment + residual axes)",
        "CR (%)": m["cumulative_return"],
        "Sharpe": m["sharpe_ratio"],
        "Sortino": m["sortino_ratio"],
        "MDD (%)": m["max_drawdown_pct"],
        "Calmar": m["calmar_ratio"],
    })
    print(f"SRF (sentiment + residual axes): CR={m['cumulative_return']:.2f}% Sharpe={m['sharpe_ratio']:.3f}")

    temperature, temp_results = select_conviction_temperature(train)
    temp_results.to_csv("backtest_results/semantic_conviction_validation.csv", index=False)
    weights, intercept, weights_df = fit_semantic_weights(
        train,
        signal_cols=SIGNAL_COLS,
        strategy="SCW (conviction-weighted)",
    )
    weights_df["intercept"] = intercept
    weights_df["temperature"] = temperature
    all_weights.append(weights_df)
    scores = predict_scores(test, weights, intercept, SIGNAL_COLS)
    curve = run_conviction_weighted_portfolio(scores, temperature=temperature)
    curve["strategy"] = "SCW (conviction-weighted)"
    all_curves.append(curve)
    m = compute_full_metrics(curve["portfolio_value"].values, bh["portfolio_value"].values, name="SCW")
    metrics_rows.append({
        "Strategy": "SCW (conviction-weighted)",
        "CR (%)": m["cumulative_return"],
        "Sharpe": m["sharpe_ratio"],
        "Sortino": m["sortino_ratio"],
        "MDD (%)": m["max_drawdown_pct"],
        "Calmar": m["calmar_ratio"],
    })
    print(
        f"SCW (conviction-weighted, temp={temperature:g}): "
        f"CR={m['cumulative_return']:.2f}% Sharpe={m['sharpe_ratio']:.3f}"
    )

    # Keep RC-SFP as a diagnostic artifact; its first run is weaker than SFP/SCW,
    # so it is not included in the main paper table.
    scores, rc_weights, regime_trace = fit_regime_conditional_sfp(train, test)
    curve = run_topk_portfolio(scores)
    curve["strategy"] = "RC-SFP (regime-conditioned)"
    all_weights.append(rc_weights)
    curve.to_csv("backtest_results/regime_conditional_sfp.csv", index=False)
    regime_trace.to_csv("backtest_results/semantic_regime_trace.csv", index=False)
    m = compute_full_metrics(curve["portfolio_value"].values, bh["portfolio_value"].values, name="RC-SFP")
    print(f"RC-SFP diagnostic (not tabled): CR={m['cumulative_return']:.2f}% Sharpe={m['sharpe_ratio']:.3f}")

    m_bh = compute_full_metrics(bh["portfolio_value"].values, bh["portfolio_value"].values, name="Buy & Hold (EW)")
    metrics_rows.append({
        "Strategy": "Buy & Hold (EW)",
        "CR (%)": m_bh["cumulative_return"],
        "Sharpe": m_bh["sharpe_ratio"],
        "Sortino": m_bh["sortino_ratio"],
        "MDD (%)": m_bh["max_drawdown_pct"],
        "Calmar": m_bh["calmar_ratio"],
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("backtest_results/semantic_factor_portfolio_metrics.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv("backtest_results/semantic_factor_portfolio.csv", index=False)
    weights_out = pd.concat(all_weights, ignore_index=True)
    weights_out.to_csv("backtest_results/semantic_factor_weights.csv", index=False)
    plot_factor_weights(weights_out, "paper/figures/fig_semantic_factor_weights.png")
    coverage_rows = coverage_sensitivity_rows(test, sfp_scores, sfp_weights, sfp_intercept)
    pd.DataFrame(coverage_rows).to_csv("backtest_results/semantic_coverage_sensitivity.csv", index=False)
    write_coverage_table(coverage_rows, "paper/table_semantic_coverage_sensitivity.tex")
    write_latex_table(metrics_rows, "paper/table_semantic_factor_portfolio.tex")

    fig, ax = plt.subplots(figsize=(10, 4))
    for curve in all_curves:
        name = curve["strategy"].iloc[0]
        ax.plot(pd.to_datetime(curve["date"]), curve["portfolio_value"] / INITIAL_AMOUNT, label=name, lw=1.8)
    ax.plot(pd.to_datetime(bh["date"]), bh["portfolio_value"] / INITIAL_AMOUNT, label="Buy & Hold (EW)", lw=1.5, ls="--", color="#94A3B8")
    ax.set_title("Semantic Factor Portfolio Baseline (2019--2023)", fontweight="bold")
    ax.set_ylabel("Normalised portfolio value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_semantic_factor_portfolio.png", dpi=150)
    plt.close(fig)
    print("Saved semantic factor portfolio outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
