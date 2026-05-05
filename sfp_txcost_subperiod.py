#!/usr/bin/env python3
"""
SFP transaction-cost sensitivity + sub-period analysis (addresses reviewer Q4/Q5).
Uses the actual SFP weights from semantic_factor_weights.csv and the real
run_topk_portfolio / run_conviction_weighted_portfolio from semantic_factor_portfolio.py.
Outputs:
  paper/table_sfp_txcost.tex
  paper/table_sfp_subperiod.tex
  backtest_results/sfp_txcost.csv
  backtest_results/sfp_subperiod.csv
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# Import actual SFP functions
sys.path.insert(0, os.path.dirname(__file__))
from semantic_factor_portfolio import (
    run_topk_portfolio,
    run_conviction_weighted_portfolio,
    predict_scores,
    buy_hold_series,
    SIGNAL_COLS,
    INITIAL_AMOUNT,
)
from metrics_extended import compute_full_metrics

TRAIN_CSV = "train_data_multi_signal_2013_2018.csv"
TRADE_CSV = "trade_data_multi_signal_2019_2023.csv"
WEIGHTS_CSV = "backtest_results/semantic_factor_weights.csv"


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TRADE_CSV)
    for df in [train, test]:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["tic"] = df["tic"].astype(str).str.upper()
        for c in SIGNAL_COLS:
            if c not in df.columns:
                df[c] = 3.0
            df[c] = df[c].fillna(3.0)
    return train, test


def load_sfp_weights(strategy="SFP (4 factors)"):
    wdf = pd.read_csv(WEIGHTS_CSV)
    wdf = wdf[wdf["strategy"] == strategy].copy()
    weights = np.array([wdf[wdf["signal"] == c]["weight"].values[0] for c in SIGNAL_COLS])
    intercept = float(wdf["intercept"].iloc[0])
    return weights, intercept


def bah_curve(test: pd.DataFrame) -> pd.DataFrame:
    """Canonical B&H matching semantic_factor_portfolio.buy_hold_series():
    mean close price across tickers, scaled by INITIAL_AMOUNT.
    This gives the same 243.6% CR as reported in the main SFP table."""
    return buy_hold_series(test)


def metrics_from_curve(pv: np.ndarray, bah_pv: np.ndarray, name: str = "") -> dict:
    m = compute_full_metrics(pv, bah_pv, name=name)
    return {
        "CR": m["cumulative_return"],
        "Sharpe": m["sharpe_ratio"],
        "MDD": m["max_drawdown_pct"],
        "Calmar": m["calmar_ratio"],
    }


def subperiod_slice(curve: pd.DataFrame, start: str, end: str) -> np.ndarray:
    sub = curve[(curve["date"] >= start) & (curve["date"] <= end)]["portfolio_value"].values
    if len(sub) < 2:
        return np.array([1.0])
    return sub / sub[0] * INITIAL_AMOUNT


# ── Transaction-cost sweep ─────────────────────────────────────────────────────
def tx_cost_analysis(test: pd.DataFrame) -> pd.DataFrame:
    weights, intercept = load_sfp_weights("SFP (4 factors)")
    scores = predict_scores(test, weights, intercept, SIGNAL_COLS)

    bah = bah_curve(test)
    bah_pv = bah["portfolio_value"].values
    bah_m = metrics_from_curve(bah_pv, bah_pv, "B&H")

    costs = [0.0005, 0.001, 0.002, 0.005, 0.010, 0.020]
    rows = []
    for c in costs:
        curve = run_topk_portfolio(scores, cost_pct=c)
        pv = curve["portfolio_value"].values
        m = metrics_from_curve(pv, bah_pv, f"SFP_{c}")
        rows.append({
            "tx_cost_pct": c * 100,
            "SFP_CR": m["CR"], "SFP_Sharpe": m["Sharpe"], "SFP_MDD": m["MDD"],
            "BH_CR": bah_m["CR"], "BH_Sharpe": bah_m["Sharpe"],
        })
        print(f"  tx={c*100:.2f}%: SFP CR={m['CR']:.1f}% Sharpe={m['Sharpe']:.3f}  "
              f"(B&H {bah_m['CR']:.1f}% / {bah_m['Sharpe']:.3f})")
    return pd.DataFrame(rows)


def write_txcost_tex(df_tc: pd.DataFrame, out: str) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SFP transaction-cost sensitivity (2019--2023). "
        r"Each row re-evaluates the daily top-10 four-factor SFP at the stated per-trade cost. "
        r"B\&H (equal-weight price-average index, same computation as Table~2, 243.6\% CR) is cost-free throughout. "
        r"\textbf{SFP is highly turnover-sensitive}: it outperforms B\&H only at $\leq$0.1\% per trade "
        r"and underperforms at $\geq$0.2\%, collapsing at 0.5\%+. "
        r"The paper's main reported result uses 0.1\%; realistic broker costs (0.2--0.5\%) reverse the advantage. "
        r"Mirrors Table~8 (DP-PPO tx-cost sensitivity) for the direct portfolio.}",
        r"\label{tab:sfp_txcost}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"\textbf{Cost (\%)} & \textbf{SFP CR (\%)} & \textbf{SFP Sharpe} "
        r"& \textbf{SFP MDD (\%)} & \textbf{B\&H CR (\%)} & \textbf{B\&H Sharpe} \\",
        r"\midrule",
    ]
    for _, r in df_tc.iterrows():
        lines.append(
            f"{r['tx_cost_pct']:.2f} & {r['SFP_CR']:.1f} & {r['SFP_Sharpe']:.3f} "
            f"& {r['SFP_MDD']:.1f} & {r['BH_CR']:.1f} & {r['BH_Sharpe']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {out}")


# ── Sub-period analysis ────────────────────────────────────────────────────────
SUBPERIODS = [
    ("2019",             "2019-01-01", "2019-12-31"),
    ("2020",             "2020-01-01", "2020-12-31"),
    ("2021",             "2021-01-01", "2021-12-31"),
    ("2022",             "2022-01-01", "2022-12-31"),
    ("2023",             "2023-01-01", "2023-12-31"),
    ("2019--20 pre/bear","2019-01-01", "2020-03-31"),
    ("2020--21 recovery","2020-04-01", "2021-12-31"),
    ("2022--23 rate hike","2022-01-01","2023-12-31"),
]


def subperiod_analysis(test: pd.DataFrame) -> pd.DataFrame:
    weights, intercept = load_sfp_weights("SFP (4 factors)")
    scores = predict_scores(test, weights, intercept, SIGNAL_COLS)
    sfp_curve = run_topk_portfolio(scores, cost_pct=0.001)
    sfp_curve["date"] = sfp_curve["date"].astype(str)

    bah = bah_curve(test)
    bah["date"] = bah["date"].astype(str)

    rows = []
    for label, start, end in SUBPERIODS:
        sfp_sub = sfp_curve[(sfp_curve["date"] >= start) & (sfp_curve["date"] <= end)]
        bah_sub = bah[(bah["date"] >= start) & (bah["date"] <= end)]
        if len(sfp_sub) < 2:
            continue
        sfp_pv = sfp_sub["portfolio_value"].values / sfp_sub["portfolio_value"].values[0] * INITIAL_AMOUNT
        bah_pv = bah_sub["portfolio_value"].values / bah_sub["portfolio_value"].values[0] * INITIAL_AMOUNT

        sfp_m = metrics_from_curve(sfp_pv, bah_pv, "SFP")
        bah_m = metrics_from_curve(bah_pv, bah_pv, "BaH")
        excess = sfp_m["CR"] - bah_m["CR"]
        rows.append({
            "Period": label,
            "SFP_CR": sfp_m["CR"], "SFP_Sharpe": sfp_m["Sharpe"],
            "BH_CR": bah_m["CR"], "BH_Sharpe": bah_m["Sharpe"],
            "Excess_CR": excess,
        })
        sign = "+" if excess >= 0 else ""
        print(f"  {label}: SFP CR={sfp_m['CR']:.1f}% S={sfp_m['Sharpe']:.3f} | "
              f"B&H CR={bah_m['CR']:.1f}% S={bah_m['Sharpe']:.3f} | excess={sign}{excess:.1f}pp")
    return pd.DataFrame(rows)


def write_subperiod_tex(df_sp: pd.DataFrame, out: str) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SFP sub-period performance (2019--2023). "
        r"Mirrors Table~7 (DP-PPO sub-periods). Excess CR = SFP minus equal-weight B\&H (same computation as Table~2). "
        r"SFP outperforms in 3 of 5 calendar years (2020, 2021, 2023). "
        r"The single largest annual outperformance (+26pp) occurs in 2023, coinciding with the AI investment cycle rally "
        r"that disproportionately benefited the high-coverage NASDAQ names (NVDA, GOOGL, AVGO) that dominate the SFP basket; "
        r"it is not possible to rule out that SFP's 2023 excess is a disguised factor exposure rather than a semantic signal effect. "
        r"SFP underperforms in 2019 and 2022 (rate-hike, drawdown year). Cost = 0.1\% per trade.}",
        r"\label{tab:sfp_subperiod}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Period} & \textbf{SFP CR (\%)} & \textbf{SFP Sharpe} "
        r"& \textbf{B\&H CR (\%)} & \textbf{Excess CR (pp)} \\",
        r"\midrule",
    ]
    for _, r in df_sp.iterrows():
        sign = "+" if r["Excess_CR"] >= 0 else ""
        lines.append(
            f"{r['Period']} & {r['SFP_CR']:.1f} & {r['SFP_Sharpe']:.3f} "
            f"& {r['BH_CR']:.1f} & {sign}{r['Excess_CR']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {out}")


def main() -> int:
    print("Loading data...")
    _, test = load_data()

    print("\n── Transaction-cost sweep (SFP, 4-factor, actual ridge weights) ──")
    df_tc = tx_cost_analysis(test)
    df_tc.to_csv("backtest_results/sfp_txcost.csv", index=False)
    write_txcost_tex(df_tc, "paper/table_sfp_txcost.tex")

    print("\n── Sub-period analysis (SFP, 4-factor, cost=0.1%) ──")
    df_sp = subperiod_analysis(test)
    df_sp.to_csv("backtest_results/sfp_subperiod.csv", index=False)
    write_subperiod_tex(df_sp, "paper/table_sfp_subperiod.tex")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
