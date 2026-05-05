"""
Backtest SAC agent on 2019-2023 trade data and compare vs CPPO.

Answers: "Is the performance from CPPO (the algorithm) or LLM signals?"

Also writes ``paper/table_algorithm_baseline.tex`` from computed metrics so the
paper table stays aligned with the checkpoints you evaluate.

Usage:
    python backtest_sac.py
    python backtest_sac.py --sac_model trained_models/agent_sac_llm_300_ep.pth
    python backtest_sac.py --no-write-tex
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model
from env_stocktrading_multi_signal import StockTradingEnv

try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

INITIAL_AMOUNT = 1_000_000
STOCK_DIM      = 30
STATE_SPACE    = 1 + 2 * STOCK_DIM + (len(INDICATORS) + 4) * STOCK_DIM
ACTION_SPACE   = STOCK_DIM

ENV_KWARGS = dict(
    hmax=100, initial_amount=INITIAL_AMOUNT,
    num_stock_shares=[0]*STOCK_DIM,
    buy_cost_pct=[0.001]*STOCK_DIM, sell_cost_pct=[0.001]*STOCK_DIM,
    reward_scaling=1e-4, state_space=STATE_SPACE, action_space=ACTION_SPACE,
    tech_indicator_list=INDICATORS, turbulence_threshold=380,
    drawdown_penalty=0.1, make_plots=False, print_verbosity=999,
)


# ─── SAC policy loader ────────────────────────────────────────────────────────

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class SACPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net     = mlp([obs_dim] + list(hidden))
        self.mu_fc   = nn.Linear(hidden[-1], act_dim)
    def act(self, obs):
        with torch.no_grad():
            x  = self.net(obs)
            mu = self.mu_fc(x)
        return torch.tanh(mu).numpy()

def load_sac_model(path):
    ckpt    = torch.load(path, map_location="cpu")
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    policy  = SACPolicy(obs_dim, act_dim)
    # Checkpoints come from the stochastic Gaussian SAC policy and include
    # log_std_fc weights. For deterministic evaluation we only need the mean
    # path (net + mu_fc), so load those keys and ignore log_std_fc.
    policy.load_state_dict(ckpt["policy"], strict=False)
    policy.eval()
    return policy


# ─── Run backtest ─────────────────────────────────────────────────────────────

def write_algorithm_baseline_tex(
    m_cppo: dict,
    m_sac: dict,
    m_bh: dict,
    out_path: str = "paper/table_algorithm_baseline.tex",
) -> None:
    """Emit NeurIPS table rows from metrics_extended dicts (same schema as hand-written table)."""

    def row(label: str, m: dict) -> str:
        cr = float(m["cumulative_return"])
        sh = float(m["sharpe_ratio"])
        so = float(m["sortino_ratio"])
        mdd = float(m["max_drawdown_pct"])
        ca = float(m["calmar_ratio"])
        return (
            f"{label} & {cr:.3f} & {sh:.3f} & {so:.3f} & {mdd:.3f} & {ca:.3f} \\\\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm baseline with identical four-signal observations (2019--2023). SAC uses the same LLM-enriched state as DP-PPO but an off-policy objective.}",
        r"\label{tab:algorithm_baseline}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{CR (\%)} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{MDD (\%)} & \textbf{Calmar} \\",
        r"\midrule",
        row("DP-PPO (LLM signals)", m_cppo),
        row("SAC (LLM signals)", m_sac),
        row(r"Buy \& Hold (EW)", m_bh),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved → {out_path}")


def run_model(model, trade_df, unique_dates):
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    pvs  = [INITIAL_AMOUNT]
    done = False
    while not done:
        action = model.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])
    return pvs[:len(unique_dates)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cppo_model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--sac_model",  default="trained_models/agent_sac_llm_300_ep.pth")
    parser.add_argument(
        "--no-write-tex",
        action="store_true",
        help="Do not regenerate paper/table_algorithm_baseline.tex.",
    )
    parser.add_argument("--tex_out", default="paper/table_algorithm_baseline.tex")
    args = parser.parse_args()

    if not os.path.exists(args.sac_model):
        print(f"SAC model not found: {args.sac_model}")
        print("Run: python train_sac_baseline.py --local_data train_data_multi_signal_2013_2018.csv")
        return

    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    n = len(unique_dates)

    bh = INITIAL_AMOUNT * (
        trade.reset_index().groupby("date")["close"].mean().sort_index().values /
        trade.reset_index().groupby("date")["close"].mean().sort_index().values[0]
    )[:n]

    print("Loading models …")
    cppo = load_cppo_model(args.cppo_model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)
    sac  = load_sac_model(args.sac_model)

    print("Running CPPO …")
    pvs_cppo = run_model(cppo, trade, unique_dates)
    print("Running SAC …")
    pvs_sac  = run_model(sac,  trade, unique_dates)

    m_cppo = compute_full_metrics(pvs_cppo, bh, name="CPPO (LLM signals)")
    m_sac  = compute_full_metrics(pvs_sac,  bh, name="SAC  (LLM signals)")
    m_bh   = compute_full_metrics(bh,        bh, name="Buy & Hold (EW)")

    print_metrics(m_cppo)
    print_metrics(m_sac)
    print_metrics(m_bh)

    # Algorithm comparison table
    print(f"\n{'═'*65}")
    print("  Algorithm Comparison: CPPO vs SAC (same 4 LLM signals)")
    print(f"{'═'*65}")
    for k in ["cumulative_return","annual_return","sharpe_ratio","sortino_ratio",
              "max_drawdown_pct","rachev_ratio","calmar_ratio"]:
        v_cppo = m_cppo.get(k, "—")
        v_sac  = m_sac.get(k, "—")
        v_bh   = m_bh.get(k, "—")
        label  = k.replace("_"," ").title()
        print(f"  {label:<26}: CPPO={v_cppo:>8}  SAC={v_sac:>8}  B&H={v_bh:>8}")

    # Save
    pd.DataFrame([m_cppo, m_sac, m_bh]).to_csv(
        "backtest_results/algorithm_comparison.csv", index=False
    )
    print("\nSaved → backtest_results/algorithm_comparison.csv")

    # Plot
    dates_dt  = pd.to_datetime(unique_dates[:n])
    fig, ax   = plt.subplots(figsize=(12, 5))
    ax.plot(dates_dt, np.array(pvs_cppo)/pvs_cppo[0], color="#2563EB", lw=2.2, label="CPPO (LLM)")
    ax.plot(dates_dt, np.array(pvs_sac)/pvs_sac[0],   color="#10B981", lw=1.8, ls="--", label="SAC (LLM)")
    ax.plot(dates_dt, bh[:n]/bh[0],                    color="#94A3B8", lw=1.5, ls=":", label="Buy & Hold (EW)")
    ax.set_title("Algorithm Comparison: CPPO vs SAC (same 4 LLM signals)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Portfolio Value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig("paper/figures/fig_algorithm_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved → paper/figures/fig_algorithm_comparison.png")

    if not args.no_write_tex:
        write_algorithm_baseline_tex(m_cppo, m_sac, m_bh, out_path=args.tex_out)


if __name__ == "__main__":
    main()
