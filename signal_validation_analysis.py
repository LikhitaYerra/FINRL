"""
Signal validation analysis for paper section.
Produces:
  - paper/figures/fig_signal_validation.png  (2x2: distributions + correlation heatmap + temporal stability + score-return alignment)
  - backtest_results/signal_validation.json  (numbers cited in paper)
No new API calls — uses existing scored data.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv("train_data_multi_signal_2013_2018.csv")
test  = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
df    = pd.concat([train, test], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

SIG = ["llm_sentiment", "llm_risk", "llm_confidence", "llm_volatility_forecast"]
LABELS = ["Sentiment", "Risk", "Confidence", "Vol. Forecast"]

# Non-neutral mask
nonneutral = (df[SIG] != 3.0).any(axis=1)
df_nn = df[nonneutral].copy()

# ── 1. Inter-signal Spearman correlation (non-neutral days only) ─────────────
corr_matrix = df_nn[SIG].corr(method="spearman")

# ── 2. Score distributions (non-neutral days) ────────────────────────────────
dist_stats = {}
for col, lab in zip(SIG, LABELS):
    vals = df_nn[col][df_nn[col] != 3.0].dropna()
    dist_stats[lab] = {
        "mean": round(float(vals.mean()), 3),
        "std":  round(float(vals.std()), 3),
        "pct_above_neutral": round(float((vals > 3).mean() * 100), 1),
        "pct_below_neutral": round(float((vals < 3).mean() * 100), 1),
    }

# ── 3. Temporal stability — lag-1 autocorrelation per ticker ─────────────────
autocorrs = {lab: [] for lab in LABELS}
for tic, gdf in df.groupby("tic"):
    gdf = gdf.sort_values("date")
    for col, lab in zip(SIG, LABELS):
        s = gdf[col].fillna(3.0)
        if s.std() > 0 and len(s) > 20:
            r, _ = stats.pearsonr(s[:-1], s[1:])
            autocorrs[lab].append(r)

autocorr_means = {lab: round(float(np.mean(v)), 3) for lab, v in autocorrs.items()}
autocorr_stds  = {lab: round(float(np.std(v)), 3)  for lab, v in autocorrs.items()}

# ── 4. Score-return alignment: mean 5-day fwd return by score quintile ───────
df_aligned = df.copy()
df_aligned = df_aligned.sort_values(["tic", "date"])
df_aligned["fwd5"] = df_aligned.groupby("tic")["close"].pct_change(5).shift(-5) * 100

quintile_alignment = {}
for col, lab in zip(SIG, LABELS):
    sub = df_aligned[df_aligned[col] != 3.0][["date", col, "fwd5"]].dropna()
    if len(sub) < 50:
        continue
    sub["quintile"] = pd.qcut(sub[col], q=5, labels=False, duplicates="drop")
    q_means = sub.groupby("quintile")["fwd5"].mean().to_dict()
    # IC: rank correlation between signal and 5-day fwd return
    ic, p = stats.spearmanr(sub[col], sub["fwd5"])
    quintile_alignment[lab] = {
        "ic": round(float(ic), 4),
        "ic_pval": round(float(p), 4),
        "quintile_means": {int(k): round(float(v), 3) for k, v in q_means.items()},
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

# Panel A: score distributions (violin)
ax_a = fig.add_subplot(gs[0, 0])
plot_data = [df_nn[col][df_nn[col] != 3.0].dropna().values for col in SIG]
vp = ax_a.violinplot(plot_data, positions=range(1, 5), showmedians=True, showextrema=False)
for i, pc in enumerate(vp["bodies"]):
    pc.set_facecolor(COLORS[i])
    pc.set_alpha(0.7)
vp["cmedians"].set_color("black")
ax_a.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, label="Neutral (3.0)")
ax_a.set_xticks(range(1, 5))
ax_a.set_xticklabels(["Sent.", "Risk", "Conf.", "Vol."], fontsize=9)
ax_a.set_ylabel("Score", fontsize=9)
ax_a.set_title("(A) Score distributions\n(non-neutral days only)", fontsize=9)
ax_a.legend(fontsize=7)
ax_a.set_ylim(1, 5)

# Panel B: Spearman correlation heatmap
ax_b = fig.add_subplot(gs[0, 1])
cm = corr_matrix.values
im = ax_b.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax_b.set_xticks(range(4))
ax_b.set_yticks(range(4))
ax_b.set_xticklabels(["Sent.", "Risk", "Conf.", "Vol."], fontsize=8, rotation=30)
ax_b.set_yticklabels(["Sent.", "Risk", "Conf.", "Vol."], fontsize=8)
for i in range(4):
    for j in range(4):
        ax_b.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8,
                  color="white" if abs(cm[i,j]) > 0.5 else "black")
plt.colorbar(im, ax=ax_b, shrink=0.8)
ax_b.set_title("(B) Inter-signal Spearman\ncorrelation (non-neutral days)", fontsize=9)

# Panel C: temporal autocorrelation
ax_c = fig.add_subplot(gs[1, 0])
labs = list(autocorr_means.keys())
means = [autocorr_means[l] for l in labs]
errs  = [autocorr_stds[l]  for l in labs]
bars = ax_c.bar(labs, means, yerr=errs, color=COLORS, alpha=0.8, capsize=4)
ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_c.set_ylabel("Lag-1 autocorrelation", fontsize=9)
ax_c.set_title("(C) Temporal stability\n(lag-1 autocorr. per ticker, mean±std)", fontsize=9)
ax_c.set_xticklabels(["Sent.", "Risk", "Conf.", "Vol."], fontsize=9)
ax_c.set_ylim(-0.1, 0.4)

# Panel D: IC bar chart (signal vs 5-day fwd return)
ax_d = fig.add_subplot(gs[1, 1])
ic_labs  = list(quintile_alignment.keys())
ic_vals  = [quintile_alignment[l]["ic"] for l in ic_labs]
ic_pvs   = [quintile_alignment[l]["ic_pval"] for l in ic_labs]
bar_colors = [COLORS[LABELS.index(l)] for l in ic_labs]
bars2 = ax_d.bar(ic_labs, ic_vals, color=bar_colors, alpha=0.8)
for bar, pv, ic in zip(bars2, ic_pvs, ic_vals):
    star = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
    if star:
        ax_d.text(bar.get_x() + bar.get_width()/2, ic + 0.002, star,
                  ha="center", va="bottom", fontsize=9)
ax_d.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_d.set_ylabel("Spearman IC (signal vs 5-day return)", fontsize=9)
ax_d.set_title("(D) Signal–return IC\n(* p<0.05, ** p<0.01, *** p<0.001)", fontsize=9)
ax_d.set_xticklabels(["Sent.", "Risk", "Conf.", "Vol."], fontsize=8)

fig.suptitle("LLM Semantic Signal Validation", fontsize=11, fontweight="bold", y=1.01)

out_path = "paper/figures/fig_signal_validation.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figure → {out_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "n_nonneutral_stockdays": int(nonneutral.sum()),
    "total_stockdays": int(len(df)),
    "coverage_pct": round(float(nonneutral.mean() * 100), 2),
    "score_distributions": dist_stats,
    "spearman_correlation": {
        f"{LABELS[i]}_vs_{LABELS[j]}": round(float(corr_matrix.iloc[i, j]), 3)
        for i in range(4) for j in range(i+1, 4)
    },
    "temporal_autocorrelation_lag1": {
        lab: {"mean": autocorr_means[lab], "std": autocorr_stds[lab]}
        for lab in LABELS
    },
    "signal_return_ic": quintile_alignment,
}
with open("backtest_results/signal_validation.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved → backtest_results/signal_validation.json")

# Print summary for paper text
print("\n── Summary for paper ──")
print(f"Total stock-days: {len(df):,}  |  Non-neutral: {nonneutral.sum():,} ({100*nonneutral.mean():.1f}%)")
print("\nScore distributions (non-neutral days):")
for lab, s in dist_stats.items():
    print(f"  {lab}: mean={s['mean']}, above-neutral={s['pct_above_neutral']}%, below-neutral={s['pct_below_neutral']}%")
print("\nTemporal lag-1 autocorrelation (mean across tickers):")
for lab in LABELS:
    print(f"  {lab}: {autocorr_means[lab]:.3f} ± {autocorr_stds[lab]:.3f}")
print("\nSignal–return IC (full panel):")
for lab, v in quintile_alignment.items():
    print(f"  {lab}: IC={v['ic']:.4f}  p={v['ic_pval']:.4f}")
print("\nInter-signal correlations (top pairs):")
for k, v in result["spearman_correlation"].items():
    print(f"  {k}: {v}")
