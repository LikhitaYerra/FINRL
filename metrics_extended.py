"""
Extended risk/performance metrics for paper-quality evaluation.

Implements:
  - Rachev Ratio  (CVaR_alpha(R+) / CVaR_alpha(R-))   [FinRL Contest metric]
  - CVaR at 1%, 5%
  - Omega Ratio
  - Max Drawdown Duration (trading days)
  - Outperformance Frequency (overall and in bear-market regime)
  - Annualised Volatility
  - Wilcoxon signed-rank test between two return series

All functions accept plain Python lists or numpy arrays of portfolio values.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """Daily log returns from portfolio value series."""
    pv = np.asarray(portfolio_values, dtype=float)
    return np.diff(np.log(pv + 1e-9))


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional Value-at-Risk at level alpha (Expected Shortfall).
    Returns the mean of the worst alpha-fraction of daily returns.
    Negative number = expected loss in tail.
    """
    sorted_r = np.sort(returns)
    cutoff    = int(np.ceil(alpha * len(sorted_r)))
    cutoff    = max(cutoff, 1)
    return float(sorted_r[:cutoff].mean())


def rachev_ratio(
    returns: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> float:
    """
    Rachev Ratio = CVaR_beta(upper tail) / |CVaR_alpha(lower tail)|

    A ratio > 1 means upside tail is larger than downside tail.
    Uses alpha=beta=5% by default (standard FinRL Contest setting).
    """
    sorted_r = np.sort(returns)
    n        = len(sorted_r)
    # Lower tail
    lower_cut = max(int(np.ceil(alpha * n)), 1)
    cvar_neg  = abs(sorted_r[:lower_cut].mean())
    # Upper tail
    upper_cut = max(int(np.ceil(beta * n)), 1)
    cvar_pos  = sorted_r[-upper_cut:].mean()

    return float(cvar_pos / (cvar_neg + 1e-9))


def max_drawdown(portfolio_values: np.ndarray) -> tuple[float, int]:
    """
    Returns (max_drawdown_pct, max_drawdown_duration_days).
    Duration = longest consecutive period below running peak.
    """
    pv     = np.asarray(portfolio_values, dtype=float)
    peak   = np.maximum.accumulate(pv)
    dd_pct = (pv - peak) / peak * 100

    # Duration: count consecutive days in drawdown
    in_dd  = (pv < peak).astype(int)
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return float(dd_pct.min()), max_dur


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio = sum(gains above threshold) / sum(losses below threshold).
    """
    gains  = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    return float(gains.sum() / (losses.sum() + 1e-9))


def outperformance_frequency(
    agent_values: np.ndarray,
    benchmark_values: np.ndarray,
    regime_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Fraction of days where daily agent return > daily benchmark return.
    If regime_mask provided (bool array), restrict to those days.
    """
    a = np.asarray(agent_values, dtype=float)
    b = np.asarray(benchmark_values, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    agent_ret = np.diff(a) / a[:-1]
    bench_ret = np.diff(b) / b[:-1]
    beats     = (agent_ret > bench_ret).astype(float)

    if regime_mask is not None:
        mask = np.asarray(regime_mask, dtype=bool)[:len(beats)]
        beats = beats[mask]

    return float(beats.mean()) * 100 if len(beats) > 0 else 0.0


def bear_market_mask(portfolio_values: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Simple bear-market detection: below rolling moving average.
    Returns bool array aligned with portfolio_values[1:] (daily returns).
    Window is capped to min(60, len//2) to handle short sub-periods.
    """
    pv     = np.asarray(portfolio_values, dtype=float)
    w      = min(window, max(1, len(pv) // 2))
    ma     = np.convolve(pv, np.ones(w) / w, mode='full')[:len(pv)]
    below  = (pv < ma)
    return below[1:]   # align with returns


def wilcoxon_test(
    agent_values: np.ndarray,
    benchmark_values: np.ndarray,
) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test: H0 = agent and benchmark have same distribution.
    Returns (statistic, p_value). p < 0.05 → statistically significant difference.
    """
    a = np.asarray(agent_values, dtype=float)
    b = np.asarray(benchmark_values, dtype=float)
    n = min(len(a), len(b))

    agent_ret = np.diff(a[:n]) / a[:n-1]
    bench_ret = np.diff(b[:n]) / b[:n-1]
    diff      = agent_ret - bench_ret

    try:
        stat, pval = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        stat, pval = 0.0, 1.0

    return float(stat), float(pval)


# ─────────────────────────────────────────────────────────────────────────────
# Master metrics function
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_metrics(
    portfolio_values: list | np.ndarray,
    benchmark_values: Optional[list | np.ndarray] = None,
    name: str = "Agent",
    initial_capital: float = 1_000_000,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute the full set of metrics used in the paper.
    
    Args:
        portfolio_values:  Daily portfolio values (list or array)
        benchmark_values:  Optional benchmark for outperformance metrics
        name:              Label for this strategy
        
    Returns:
        dict with all metrics
    """
    pv   = np.asarray(portfolio_values, dtype=float)
    ret  = np.diff(pv) / pv[:-1]
    lret = to_returns(pv)

    # Basic metrics
    cumulative_return  = (pv[-1] / pv[0] - 1) * 100
    annual_return      = ((pv[-1] / pv[0]) ** (trading_days_per_year / len(pv)) - 1) * 100
    volatility         = ret.std() * np.sqrt(trading_days_per_year) * 100
    sharpe             = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(trading_days_per_year)

    neg_ret  = ret[ret < 0]
    sortino  = (ret.mean() / (neg_ret.std() + 1e-9)) * np.sqrt(trading_days_per_year)

    mdd_pct, mdd_dur   = max_drawdown(pv)
    calmar             = annual_return / abs(mdd_pct + 1e-9)

    # Advanced metrics
    cvar_1  = cvar(ret, alpha=0.01) * 100
    cvar_5  = cvar(ret, alpha=0.05) * 100
    rr      = rachev_ratio(ret)
    omega   = omega_ratio(ret)

    result = dict(
        name               = name,
        final_value        = round(float(pv[-1]), 2),
        cumulative_return  = round(cumulative_return, 3),
        annual_return      = round(annual_return, 3),
        volatility         = round(volatility, 3),
        sharpe_ratio       = round(sharpe, 4),
        sortino_ratio      = round(sortino, 4),
        max_drawdown_pct   = round(mdd_pct, 3),
        max_dd_duration_d  = mdd_dur,
        calmar_ratio       = round(calmar, 4),
        cvar_1pct          = round(cvar_1, 4),
        cvar_5pct          = round(cvar_5, 4),
        rachev_ratio       = round(rr, 4),
        omega_ratio        = round(omega, 4),
        n_trading_days     = len(pv),
    )

    # Outperformance vs benchmark
    if benchmark_values is not None:
        bv = np.asarray(benchmark_values, dtype=float)
        bear_mask = bear_market_mask(bv[:len(pv)])
        result["outperf_overall"]    = round(outperformance_frequency(pv, bv[:len(pv)]), 2)
        result["outperf_bear"]       = round(outperformance_frequency(pv, bv[:len(pv)], regime_mask=bear_mask), 2)
        wstat, wpval                 = wilcoxon_test(pv, bv[:len(pv)])
        result["wilcoxon_stat"]      = round(wstat, 2)
        result["wilcoxon_pval"]      = round(wpval, 4)
        result["significant_5pct"]   = wpval < 0.05

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(m: dict):
    """Print a metrics dict in a readable format."""
    WIDTH = 26
    print(f"\n{'─'*50}")
    print(f"  {m['name']}")
    print(f"{'─'*50}")

    groups = [
        ("Return",       ["cumulative_return", "annual_return", "volatility"]),
        ("Risk-Adj",     ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]),
        ("Drawdown",     ["max_drawdown_pct", "max_dd_duration_d"]),
        ("Tail Risk",    ["cvar_1pct", "cvar_5pct", "rachev_ratio", "omega_ratio"]),
        ("vs Benchmark", ["outperf_overall", "outperf_bear", "wilcoxon_pval", "significant_5pct"]),
    ]
    for group_name, keys in groups:
        printed = False
        for k in keys:
            if k in m:
                if not printed:
                    print(f"\n  [{group_name}]")
                    printed = True
                label = k.replace("_", " ").title()
                print(f"    {label:<{WIDTH}}: {m[k]}")


if __name__ == "__main__":
    import json

    # Quick test with real backtest data
    import pandas as pd
    df = pd.read_csv("backtest_results/portfolio_value.csv")

    agent_pv = df["cppo_value"].values
    bh_pv    = df["bh_value"].values

    agent_m = compute_full_metrics(agent_pv, bh_pv, name="CPPO (LLM signals)")
    bh_m    = compute_full_metrics(bh_pv, bh_pv, name="Buy & Hold (EW)")

    print_metrics(agent_m)
    print_metrics(bh_m)

    print(f"\n  Rachev Ratio (agent): {agent_m['rachev_ratio']:.4f}")
    print(f"  Rachev Ratio (B&H):   {bh_m['rachev_ratio']:.4f}")
    print(f"  Statistically significant vs B&H: {agent_m.get('significant_5pct')}")
