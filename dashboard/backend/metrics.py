"""Shared metric computation used by the API."""

import numpy as np
import pandas as pd


def cumulative_return(series: list[float]) -> float:
    vals = np.array(series, dtype=float)
    return float((vals[-1] - vals[0]) / vals[0]) if vals[0] != 0 else 0.0


def max_drawdown(series: list[float]) -> float:
    vals = np.array(series, dtype=float)
    peak = np.maximum.accumulate(vals)
    dd = (peak - vals) / np.where(peak == 0, 1e-9, peak)
    return float(np.max(dd))


def rachev_ratio(series: list[float], tail_pct: float = 5.0) -> float | None:
    vals = np.array(series, dtype=float)
    if len(vals) < 10:
        return None
    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1e-9, vals[:-1])
    upper = np.percentile(rets, 100 - tail_pct)
    lower = np.percentile(rets, tail_pct)
    upside = rets[rets >= upper]
    downside = rets[rets <= lower]
    e_up = float(np.mean(upside)) if len(upside) else 0.0
    e_dn = float(np.mean(-downside)) if len(downside) else 1e-9
    return round(e_up / e_dn, 4) if e_dn != 0 else None


def sharpe_ratio(series: list[float], ann_factor: float = 252) -> float | None:
    vals = np.array(series, dtype=float)
    if len(vals) < 2:
        return None
    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1e-9, vals[:-1])
    std = float(np.std(rets))
    if std == 0:
        return None
    return round(float(np.mean(rets)) / std * (ann_factor ** 0.5), 4)


def outperformance_frequency(
    portfolio: list[float],
    benchmark: list[float],
) -> dict:
    n = min(len(portfolio), len(benchmark))
    p = np.array(portfolio[:n], dtype=float)
    b = np.array(benchmark[:n], dtype=float)
    p_ret = np.diff(p) / np.where(p[:-1] == 0, 1e-9, p[:-1])
    b_ret = np.diff(b) / np.where(b[:-1] == 0, 1e-9, b[:-1])
    overall = float((p_ret > b_ret).mean())
    bear_mask = b_ret < -0.01
    bear_out = float((p_ret[bear_mask] > b_ret[bear_mask]).mean()) if bear_mask.sum() > 0 else None
    return {"overall": round(overall, 4), "during_downturns": round(bear_out, 4) if bear_out is not None else None}


def compute_all_metrics(portfolio: list[float], benchmark: list[float], name: str = "") -> dict:
    out_freq = outperformance_frequency(portfolio, benchmark)
    return {
        "name": name,
        "cumulative_return": round(cumulative_return(portfolio) * 100, 2),
        "max_drawdown": round(max_drawdown(portfolio) * 100, 2),
        "rachev_ratio": rachev_ratio(portfolio),
        "sharpe_ratio": sharpe_ratio(portfolio),
        "outperform_freq_overall": round(out_freq["overall"] * 100, 1) if out_freq["overall"] is not None else None,
        "outperform_freq_downturns": round(out_freq["during_downturns"] * 100, 1) if out_freq["during_downturns"] is not None else None,
    }
