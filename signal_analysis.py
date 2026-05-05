"""
LLM Signal Quality Analysis

Answers the key research questions about the 4 LLM signals:

  Q1. Do LLM signals predict future returns?
      → Computes Spearman correlation between signal and N-day forward returns
      → For each signal, lag (1,3,5,10,20 days), and stock

  Q2. How fast do signals decay?
      → Predictive IC (Information Coefficient) vs lag
      → Shows optimal signal freshness window

  Q3. Are the 4 signals complementary (low mutual correlation)?
      → Signal-signal correlation matrix
      → Confirms they capture different information

  Q4. Do signals differ across market regimes?
      → Mean signal value in bull vs bear markets

  Q5. What is the signal coverage / null rate?
      → % of stock-dates with real vs neutral (imputed) signals

Outputs:
  paper/figures/fig4_signal_ic.png           (IC decay curve)
  paper/figures/fig5_signal_correlation.png  (signal-signal heatmap)
  paper/figures/fig6_signal_by_regime.png    (per-regime signal distribution)
  backtest_results/signal_analysis.json      (all numeric results)

Usage:
    python signal_analysis.py
"""

from __future__ import annotations

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ───────────────────────────────────────────────────────────────────

SIGNAL_NAMES = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
SIGNAL_LABELS = ["Sentiment", "Risk", "Confidence", "Volatility Forecast"]
LAGS         = [1, 3, 5, 10, 20]       # forward return lags in trading days
FIG_DIR      = "paper/figures"
DATA_DIR     = "backtest_results"


# ─── Information Coefficient ──────────────────────────────────────────────────

def compute_ic_by_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (signal, lag) pair, compute the Spearman IC across all
    (stock × date) observations.

    IC = Spearman corr between signal today and stock return N days later.
    """
    records = []
    pivot_close = df.pivot(index="date", columns="tic", values="close").sort_index()
    dates       = sorted(df["date"].unique())

    for sig, label in zip(SIGNAL_NAMES, SIGNAL_LABELS):
        pivot_sig = df.pivot(index="date", columns="tic", values=sig).sort_index()
        pivot_sig = pivot_sig.reindex(dates).fillna(3.0)  # neutral fill

        for lag in LAGS:
            # Forward return: close[t+lag] / close[t] - 1
            fwd_ret = pivot_close.pct_change(lag).shift(-lag)

            # Stack both into flat vectors aligned by (date, tic)
            sig_flat = pivot_sig.values.flatten()
            ret_flat = fwd_ret.values.flatten()

            # Remove NaN pairs
            mask = np.isfinite(sig_flat) & np.isfinite(ret_flat)
            s, r = sig_flat[mask], ret_flat[mask]

            if len(s) < 30:
                ic, pval = np.nan, np.nan
            else:
                ic, pval = stats.spearmanr(s, r)

            records.append({
                "signal": label,
                "lag":    lag,
                "ic":     float(ic),
                "pval":   float(pval),
                "n":      int(mask.sum()),
                "significant": bool(pval < 0.05) if not np.isnan(pval) else False,
            })

    return pd.DataFrame(records)


def compute_ic_timeseries(df: pd.DataFrame, signal: str, lag: int = 5) -> pd.Series:
    """
    Compute monthly rolling IC for a specific signal.
    Shows if predictive power is stable or varies over time.
    """
    pivot_close = df.pivot(index="date", columns="tic", values="close").sort_index()
    pivot_sig   = df.pivot(index="date", columns="tic", values=signal).sort_index().fillna(3.0)
    dates       = sorted(pivot_close.index)

    monthly_dates = pd.date_range(
        start=pd.to_datetime(dates[0]),
        end=pd.to_datetime(dates[-1]),
        freq="ME",
    )

    fwd_ret = pivot_close.pct_change(lag).shift(-lag)
    ics     = {}

    for i, dt in enumerate(monthly_dates[:-1]):
        window_start = dt - pd.DateOffset(months=1)
        # Find dates in window
        window_dates = [d for d in dates
                        if str(window_start.date()) <= d <= str(dt.date())]
        if len(window_dates) < 5:
            continue

        sig_vals = pivot_sig.loc[pivot_sig.index.isin(window_dates)].values.flatten()
        ret_vals = fwd_ret.loc[fwd_ret.index.isin(window_dates)].values.flatten()
        mask     = np.isfinite(sig_vals) & np.isfinite(ret_vals)
        if mask.sum() < 20:
            continue
        ic, _ = stats.spearmanr(sig_vals[mask], ret_vals[mask])
        ics[dt] = ic

    return pd.Series(ics)


# ─── Signal-signal correlation ────────────────────────────────────────────────

def signal_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation matrix of the 4 signals across all (stock, date).
    Low off-diagonal values → signals are complementary (good for ensemble).
    """
    sig_data = {}
    for sig, label in zip(SIGNAL_NAMES, SIGNAL_LABELS):
        vals = df[sig].fillna(3.0).values
        sig_data[label] = vals

    sig_df = pd.DataFrame(sig_data)
    return sig_df.corr(method="spearman")


# ─── Signal coverage ──────────────────────────────────────────────────────────

def signal_coverage(df: pd.DataFrame) -> dict:
    """
    Fraction of (stock, date) entries that have real (non-neutral) signals.
    A score == 3.0 exactly indicates imputed neutral.
    """
    coverage = {}
    n_total = len(df)
    for sig, label in zip(SIGNAL_NAMES, SIGNAL_LABELS):
        if sig not in df.columns:
            coverage[label] = 0.0
            continue
        # Neutral = exactly 3.0 (imputed default)
        non_neutral = (df[sig].fillna(3.0) != 3.0).sum()
        coverage[label] = float(non_neutral / n_total * 100)
    return coverage


# ─── Signal by regime ──────────────────────────────────────────────────────────

def signal_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split into bull/bear based on 60-day SMA of mean close price.
    Compute mean signal value per regime.
    """
    # Compute market return as mean of all stocks
    mkt = df.groupby("date")["close"].mean().sort_index()
    sma = mkt.rolling(60, min_periods=1).mean()
    regime_map = {
        date: ("Bull" if mkt.loc[date] > sma.loc[date] else "Bear")
        for date in mkt.index
    }
    df["regime"] = df["date"].map(regime_map)

    rows = []
    for regime in ["Bull", "Bear"]:
        subset = df[df["regime"] == regime]
        for sig, label in zip(SIGNAL_NAMES, SIGNAL_LABELS):
            vals = subset[sig].fillna(3.0)
            rows.append({
                "Signal": label,
                "Regime": regime,
                "Mean":   float(vals.mean()),
                "Std":    float(vals.std()),
                "N":      len(vals),
            })
    return pd.DataFrame(rows)


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_ic_decay(ic_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ["#2563EB", "#EF4444", "#10B981", "#F59E0B"]
    for (label,), group in ic_df.groupby(["signal"]):
        color = palette[SIGNAL_LABELS.index(label) % len(palette)]
        ax.plot(group["lag"], group["ic"] * 100, marker="o", label=label,
                color=color, linewidth=2)
        ax.fill_between(group["lag"],
                        (group["ic"] - 0.005) * 100,
                        (group["ic"] + 0.005) * 100,
                        alpha=0.1, color=color)

    ax.axhline(0, color="black", linewidth=0.8, ls="--")
    ax.set_xlabel("Forward Return Lag (trading days)", fontsize=11)
    ax.set_ylabel("Spearman IC × 100", fontsize=11)
    ax.set_title("LLM Signal Predictive Information Coefficient vs Return Lag",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(LAGS)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 4 → {out}")


def plot_signal_correlation(corr_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.zeros_like(corr_df.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_df, ax=ax, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, mask=mask,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("LLM Signal Inter-Correlation (Spearman)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 5 → {out}")


def plot_signal_by_regime(regime_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    x      = np.arange(len(SIGNAL_LABELS))
    w      = 0.35
    bull   = regime_df[regime_df["Regime"] == "Bull"].set_index("Signal")
    bear   = regime_df[regime_df["Regime"] == "Bear"].set_index("Signal")

    ax.bar(x - w/2, [bull.loc[l, "Mean"] for l in SIGNAL_LABELS], w,
           label="Bull", color="#10B981", alpha=0.8)
    ax.bar(x + w/2, [bear.loc[l, "Mean"] for l in SIGNAL_LABELS], w,
           label="Bear", color="#EF4444", alpha=0.8)

    ax.axhline(3.0, color="black", linewidth=0.8, ls="--", label="Neutral (3.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_LABELS, fontsize=10)
    ax.set_ylabel("Mean LLM Signal Score (1–5 scale)", fontsize=10)
    ax.set_title("LLM Signal Values by Market Regime (Bull vs Bear)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(1, 5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 6 → {out}")


def plot_ic_timeseries(df: pd.DataFrame, out: str):
    """Plot rolling monthly IC over the full 2013-2023 period (train + trade)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    palette = ["#2563EB", "#EF4444", "#10B981", "#F59E0B"]

    for i, (sig, label) in enumerate(zip(SIGNAL_NAMES, SIGNAL_LABELS)):
        ts = compute_ic_timeseries(df, sig, lag=5)
        if ts.empty:
            continue
        ax = axes[i]
        ax.bar(ts.index, ts.values * 100, color=palette[i], alpha=0.6, width=20)
        ax.plot(ts.index, ts.rolling(3, min_periods=1).mean() * 100,
                color=palette[i], linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8, ls="--")
        ax.set_title(f"{label} — 5-day IC (monthly)", fontweight="bold", fontsize=10)
        ax.set_ylabel("IC × 100")
        ax.grid(alpha=0.3)

    fig.suptitle("LLM Signal Information Coefficient over Time (2013–2023)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  FIGURE 7 → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load both train and trade data for full coverage
    print("Loading signal data …")
    frames = []
    for f in ["train_data_multi_signal_2013_2018.csv",
              "trade_data_multi_signal_2019_2023.csv"]:
        if os.path.exists(f):
            df_ = pd.read_csv(f)
            frames.append(df_)
    if not frames:
        print("ERROR: No signal data found.")
        return
    df = pd.concat(frames, ignore_index=True)
    for sig in SIGNAL_NAMES:
        if sig in df.columns:
            df[sig] = pd.to_numeric(df[sig], errors="coerce")

    print(f"  Total rows: {len(df):,}  |  Stocks: {df['tic'].nunique()}"
          f"  |  Dates: {df['date'].nunique()}")

    # ── Signal coverage ────────────────────────────────────────────────────────
    print("\n[1] Signal coverage …")
    cov = signal_coverage(df)
    for label, pct in cov.items():
        print(f"  {label:<22}: {pct:.1f}% non-neutral")

    # ── Signal-signal correlation ──────────────────────────────────────────────
    print("\n[2] Signal-signal correlation …")
    corr = signal_correlation_matrix(df)
    print(corr.round(3))

    # ── IC by lag ─────────────────────────────────────────────────────────────
    print("\n[3] IC by forward lag …")
    ic_df = compute_ic_by_lag(df)
    print(ic_df.to_string(index=False))

    # ── Signal by regime ──────────────────────────────────────────────────────
    print("\n[4] Signal values by regime …")
    regime_df = signal_by_regime(df)
    print(regime_df.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[5] Generating figures …")
    plot_ic_decay(ic_df,     f"{FIG_DIR}/fig4_signal_ic.png")
    plot_signal_correlation(corr, f"{FIG_DIR}/fig5_signal_correlation.png")
    plot_signal_by_regime(regime_df, f"{FIG_DIR}/fig6_signal_by_regime.png")
    plot_ic_timeseries(df,   f"{FIG_DIR}/fig7_ic_timeseries.png")

    # ── Save numeric results ──────────────────────────────────────────────────
    results = {
        "signal_coverage":    cov,
        "ic_by_lag":          ic_df.to_dict(orient="records"),
        "regime_comparison":  regime_df.to_dict(orient="records"),
    }
    out = f"{DATA_DIR}/signal_analysis.json"
    with open(out, "w") as f_:
        json.dump(results, f_, indent=2)
    print(f"\nSaved → {out}")

    # ── Key findings summary ───────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  Key Research Findings")
    print(f"{'═'*55}")
    print("\n  1. IC at 5-day lag:")
    for sig, label in zip(SIGNAL_NAMES, SIGNAL_LABELS):
        row = ic_df[(ic_df["signal"] == label) & (ic_df["lag"] == 5)].iloc[0]
        sig_str = "**" if row["significant"] else "  "
        print(f"  {sig_str} {label:<22}: IC={row['ic']:.4f}  p={row['pval']:.3f}"
              f"{'  ← significant' if row['significant'] else ''}")

    print("\n  2. Most complementary signal pair (lowest cross-correlation):")
    corr_vals = corr.where(~np.eye(len(corr), dtype=bool))
    min_pair  = corr_vals.stack().abs().idxmin()
    min_val   = corr_vals.loc[min_pair[0], min_pair[1]]
    print(f"     {min_pair[0]} × {min_pair[1]} : ρ={min_val:.3f}")

    print("\n  3. Bull vs Bear signal shift (sentiment):")
    bull_sent = regime_df[(regime_df["Signal"]=="Sentiment") & (regime_df["Regime"]=="Bull")]["Mean"].values[0]
    bear_sent = regime_df[(regime_df["Signal"]=="Sentiment") & (regime_df["Regime"]=="Bear")]["Mean"].values[0]
    print(f"     Bull: {bull_sent:.3f}  Bear: {bear_sent:.3f}  Δ={bull_sent-bear_sent:.3f}")


if __name__ == "__main__":
    main()
