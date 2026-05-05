"""
PC1 Loadings Analysis — answers reviewer Q1.

Computes:
1. PC1 loadings on the four SSAI axes (train period 2013-2018)
2. Softmax-weighted SFP baseline using PC1-like equal weighting
3. Comparison table: PC1-SFP vs Softmax-SFP vs 4-axis SSAI-SFP
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from metrics_extended import compute_full_metrics

AXES = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
TRAIN_CSV = "train_data_multi_signal_2013_2018.csv"
TRADE_CSV = "trade_data_multi_signal_2019_2023.csv"
TOP_N = 10

# ── 1. Fit PCA on training period ─────────────────────────────────────────────
print("Loading training data …")
train = pd.read_csv(TRAIN_CSV)
X_train = train[AXES].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=4)
pca.fit(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    index=AXES,
    columns=[f"PC{i+1}" for i in range(4)]
)
explained = pca.explained_variance_ratio_
print("\n=== PCA Explained Variance ===")
for i, ev in enumerate(explained):
    print(f"  PC{i+1}: {ev*100:.1f}%")

print("\n=== PC1 Loadings (eigenvector of first principal component) ===")
print(loadings["PC1"].to_string())
print()

# Normalised absolute loadings (shows relative contribution per axis)
abs_l = loadings["PC1"].abs()
rel = abs_l / abs_l.sum()
print("=== PC1 Loading magnitudes (normalised to sum=1) ===")
print(rel.round(3).to_string())
print()

# ── 2. Build three SFP strategies on the trade period ─────────────────────────
print("Loading trade data …")
trade = pd.read_csv(TRADE_CSV, parse_dates=["date"])

# Scale axes using training-period scaler
trade[AXES] = scaler.transform(trade[AXES])

# PC1 composite score (same as pc1_sfp_baseline.py)
pc1_weights = pca.components_[0]  # shape (4,)
trade["pc1_score"] = trade[AXES].values @ pc1_weights

# Softmax-weighted composite: equal positive weight on all axes
# (reviewer's hypothetical auditable alternative)
softmax_weights = np.ones(4) / 4.0
trade["softmax_score"] = trade[AXES].values @ softmax_weights

# Load existing 4-axis SSAI factor weights from training period
sfp_weights_path = "backtest_results/semantic_factor_weights.csv"
if os.path.exists(sfp_weights_path):
    sfp_w = pd.read_csv(sfp_weights_path)
    print("Loaded SSAI factor weights:")
    print(sfp_w.to_string(index=False))
    # Build composite SSAI score
    w_dict = dict(zip(sfp_w["signal"], sfp_w["weight"])) if "signal" in sfp_w.columns else {}
    if w_dict:
        trade["ssai_score"] = sum(
            trade[ax] * w_dict.get(ax, 0.0) for ax in AXES
        )
    else:
        # fallback: use raw mean
        trade["ssai_score"] = trade[AXES].mean(axis=1)
else:
    print("SSAI factor weights not found, using mean of axes as SSAI score")
    trade["ssai_score"] = trade[AXES].mean(axis=1)

# ── 3. Run top-N SFP for each strategy ────────────────────────────────────────
INITIAL = 1_000_000.0

def run_sfp(df, score_col, top_n=TOP_N, initial=INITIAL):
    dates = sorted(df["date"].unique())
    value = initial
    daily_values = []
    for dt in dates:
        day = df[df["date"] == dt].copy()
        # Rank by score, pick top N
        top = day.nlargest(top_n, score_col)
        # Equal-weight return for the day
        ret = top["close"].pct_change().mean() if len(top) > 1 else 0.0
        if pd.isna(ret):
            ret = 0.0
        value *= (1 + ret)
        daily_values.append({"date": dt, "value": value})
    return pd.DataFrame(daily_values)


# We need daily returns, not just close. Use close-to-close.
# Re-compute properly: for each date, take the NEXT day's return for stocks selected today.
def run_sfp_proper(df, score_col, top_n=TOP_N, initial=INITIAL):
    df = df.copy().sort_values("date")
    tickers = df["tic"].unique()
    # Pivot close prices
    prices = df.pivot(index="date", columns="tic", values="close").sort_index()
    daily_ret = prices.pct_change()  # next-day return: we use same-day close-to-close
    # Pivot scores
    scores = df.pivot(index="date", columns="tic", values=score_col)

    dates = sorted(prices.index)
    value = initial
    daily_values = []
    for i, dt in enumerate(dates):
        if i == 0:
            daily_values.append({"date": dt, "value": value})
            continue
        # Select top-N based on previous day's score
        prev_dt = dates[i - 1]
        if prev_dt in scores.index:
            sc = scores.loc[prev_dt].dropna()
            selected = sc.nlargest(top_n).index.tolist()
        else:
            selected = list(tickers[:top_n])
        # Today's returns for selected
        rets = daily_ret.loc[dt, [t for t in selected if t in daily_ret.columns]].dropna()
        ret = rets.mean() if len(rets) > 0 else 0.0
        value *= (1 + ret)
        daily_values.append({"date": dt, "value": value})
    return pd.DataFrame(daily_values)

print("\nRunning PC1-SFP …")
pc1_port = run_sfp_proper(trade, "pc1_score")
print("Running Softmax-SFP …")
sfx_port = run_sfp_proper(trade, "softmax_score")
print("Running SSAI-SFP …")
ssai_port = run_sfp_proper(trade, "ssai_score")

# Buy-and-hold equal weight
prices_all = trade.pivot(index="date", columns="tic", values="close").sort_index()
bh_ret = prices_all.pct_change().mean(axis=1)
bh_values = [INITIAL]
for r in bh_ret.iloc[1:]:
    bh_values.append(bh_values[-1] * (1 + (r if not pd.isna(r) else 0.0)))
bh_port = pd.DataFrame({"date": prices_all.index, "value": bh_values})

# ── 4. Compute metrics ─────────────────────────────────────────────────────────
def metrics_row(port, label):
    m = compute_full_metrics(port["value"].values)
    # cumulative_return is already in % (e.g. 433.6, not 4.336)
    cr = m.get("cumulative_return", m.get("total_return", 0))
    mdd = m.get("max_drawdown_pct", m.get("max_drawdown", 0))
    return {
        "Strategy": label,
        "CR (%)": round(cr, 1),
        "Sharpe": round(m.get("sharpe_ratio", m.get("sharpe", 0)), 3),
        "Sortino": round(m.get("sortino_ratio", m.get("sortino", 0)), 3),
        "MDD (%)": round(mdd, 1),
    }

rows = [
    metrics_row(pc1_port,  "PC1-SFP (data-driven)"),
    metrics_row(sfx_port,  "Softmax-SFP (equal-weight auditable)"),
    metrics_row(ssai_port, "4-axis SSAI-SFP (factor weights)"),
    metrics_row(bh_port,   "Buy & Hold (equal-weight)"),
]
results = pd.DataFrame(rows)
print("\n=== Factor Portfolio Comparison (incl. Softmax-SFP) ===")
print(results.to_string(index=False))

# ── 5. Gap decomposition ───────────────────────────────────────────────────────
pc1_cr  = results.loc[results["Strategy"].str.startswith("PC1"), "CR (%)"].values[0]
sfx_cr  = results.loc[results["Strategy"].str.startswith("Softmax"), "CR (%)"].values[0]
ssai_cr = results.loc[results["Strategy"].str.startswith("4-axis"), "CR (%)"].values[0]

gap_total   = pc1_cr - ssai_cr
gap_weighting = pc1_cr - sfx_cr     # gap due to PCA vs equal-weight scoring
gap_rule    = sfx_cr - ssai_cr      # gap due to equal-weight vs SSAI factor weights

print(f"\n=== Gap Decomposition (addressing reviewer Q1) ===")
print(f"  Total gap PC1 vs SSAI:                {gap_total:.1f} pp")
print(f"  PCA optimisation vs equal-weight:     {gap_weighting:.1f} pp  ({gap_weighting/gap_total*100:.0f}% of gap)")
print(f"  Equal-weight vs SSAI factor weights:  {gap_rule:.1f} pp  ({gap_rule/gap_total*100:.0f}% of gap)")
print()
print(f"  → Softmax-SFP achieves {sfx_cr:.1f}% CR with no interpretability loss")
print(f"    (fully auditable: equal weight on all four named axes)")

# ── 6. Save PC1 loadings table ────────────────────────────────────────────────
os.makedirs("backtest_results", exist_ok=True)
loadings_out = pd.DataFrame({
    "Axis": AXES,
    "PC1 Loading": pca.components_[0].round(4),
    "Rel. Contribution": rel.values.round(3),
})
loadings_out.to_csv("backtest_results/pc1_loadings.csv", index=False)
results.to_csv("backtest_results/factor_comparison_extended.csv", index=False)
print("Saved: backtest_results/pc1_loadings.csv")
print("Saved: backpack_results/factor_comparison_extended.csv")
