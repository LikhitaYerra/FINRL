"""
FinRL Dashboard — FastAPI backend.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET /api/portfolio     — normalised portfolio series for all agents
    GET /api/metrics       — all 4 contest metrics per agent
    GET /api/regime        — bear/bull regime timeline
    GET /api/signals       — recent LLM signal table
    GET /api/health        — health check
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from data_loader import (
    get_portfolio_data,
    get_regime_data,
    get_llm_signals,
    get_trade_data_signals,
    get_backtest_metrics,
)
from metrics import compute_all_metrics

app = FastAPI(title="FinRL LLM Trading Bot Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def portfolio(
    normalise: bool = Query(True, description="Normalise all series to start at 1.0"),
) -> dict[str, Any]:
    """
    Returns portfolio value series for each agent (in-sample 2019–2023 by default).

    When available (see ``meta.features``):
      - ``oos``: separate hold-out window from ``oos_portfolio.csv``
      - ``CPPO (5-seed μ)``: row mean from ``multi_seed_portfolio.csv``

    Response includes ``meta``: ``{mode, primary_source, features}``.
    """
    data = get_portfolio_data()
    dates: list[str] = data["dates"]
    agents: dict[str, list[float]] = data["agents"]
    meta: dict[str, Any] = data.get("meta") or {"mode": "unknown", "primary_source": None, "features": []}
    oos_block: dict[str, Any] | None = data.get("oos")

    def _norm(a: dict[str, list[float]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name, series in a.items():
            base = series[0] if series and series[0] != 0 else 1.0
            out[name] = [round(v / base, 6) for v in series]
        return out

    if normalise:
        payload: dict[str, Any] = {"dates": dates, "agents": _norm(agents), "meta": meta}
        if oos_block:
            payload["oos"] = {
                "dates": oos_block["dates"],
                "agents": _norm(oos_block["agents"]),
            }
        return payload

    out: dict[str, Any] = {"dates": dates, "agents": agents, "meta": meta}
    if oos_block:
        out["oos"] = oos_block
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Contest metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def metrics() -> list[dict]:
    """
    Returns the 4 FinRL Contest 2025 Task 1 evaluation metrics per agent:
      - cumulative_return (%)
      - max_drawdown (%)
      - rachev_ratio
      - sharpe_ratio
      - outperform_freq_overall (%)
      - outperform_freq_downturns (%)
    """
    data = get_portfolio_data()
    agents = data["agents"]

    benchmark = (
        agents.get("Buy & Hold (EW)")
        or agents.get("NASDAQ-100 (QQQ)")
        or next(iter(agents.values()))
    )

    result: list[dict] = []
    for name, series in agents.items():
        n = min(len(series), len(benchmark))
        row = compute_all_metrics(series[:n], benchmark[:n], name=name)
        result.append(row)

    result.sort(key=lambda r: r["cumulative_return"], reverse=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Regime timeline
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/regime")
def regime() -> list[dict]:
    """
    Returns [{date, regime}] where regime is 'bull' or 'bear'.
    """
    return get_regime_data()


@app.get("/api/regime/summary")
def regime_summary() -> dict:
    data = get_regime_data()
    total = len(data)
    bull = sum(1 for d in data if d["regime"] == "bull")
    bear = total - bull
    return {
        "total_days": total,
        "bull_days": bull,
        "bear_days": bear,
        "bull_pct": round(bull / total * 100, 1) if total else 0,
        "bear_pct": round(bear / total * 100, 1) if total else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM signals
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/signals")
def signals(
    limit: int = Query(200, ge=1, le=2000),
    source: str = Query("news", description="'news' for raw news scores, 'trade' for trade-period aggregates"),
) -> list[dict]:
    """
    Returns recent LLM signal rows with columns:
      date, ticker, llm_sentiment, llm_risk, llm_confidence, llm_volatility_forecast
    """
    if source == "trade":
        return get_trade_data_signals(limit=limit)
    return get_llm_signals(limit=limit)


@app.get("/api/signals/summary")
def signals_summary() -> list[dict]:
    """
    Returns per-ticker average of the 4 LLM signals (most recent 60 trading days).
    """
    rows = get_trade_data_signals(limit=6000)
    if not rows:
        return []

    import pandas as pd
    df = pd.DataFrame(rows)
    sig_cols = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
    avail = [c for c in sig_cols if c in df.columns]
    if "ticker" not in df.columns or not avail:
        return rows[:50]

    agg = df.groupby("ticker")[avail].mean().round(2).reset_index()
    return agg.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown series (derived from portfolio)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/comparison")
def model_comparison() -> list[dict]:
    """
    Returns model comparison data from backtest_results/model_comparison.csv
    (produced by compare_models.py). Falls back to empty list if not yet run.
    """
    import pandas as pd
    path = os.path.join(os.path.dirname(__file__), "..", "..", "backtest_results", "model_comparison.csv")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        return []


@app.get("/api/backtest")
def backtest_summary() -> dict:
    """
    Returns real backtest performance metrics for agent vs buy-and-hold.
    Keys: agent, buy_hold — each with cumulative_return, sharpe_ratio, etc.
    """
    return get_backtest_metrics()


@app.get("/api/drawdown")
def drawdown() -> dict:
    """
    Returns drawdown (%) series per agent.
    """
    import numpy as np

    data = get_portfolio_data()
    dates = data["dates"]
    result: dict[str, list[float]] = {}

    for name, series in data["agents"].items():
        vals = np.array(series, dtype=float)
        peak = np.maximum.accumulate(vals)
        dd = (peak - vals) / np.where(peak == 0, 1e-9, peak) * 100
        result[name] = [round(float(v), 4) for v in dd]

    return {"dates": dates, "agents": result}
