"""
Backtest SAM-PPO (standalone) vs plain CPPO vs Buy & Hold.

Answers: "Does the signal attention module improve on the plain MLP?"
A positive result means the cross-attention mechanism for LLM signals
adds genuine predictive value beyond simple feature concatenation.

Usage:
    python backtest_sam.py
    python backtest_sam.py --sam_model trained_models/agent_sam_ppo_30e.pth
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from metrics_extended import compute_full_metrics, print_metrics
from model_loader import load_cppo_model, load_sam_mpi_cppo
from signal_attention import SAMStateEncoder
from env_stocktrading_multi_signal import StockTradingEnv

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


# ─── SAM model class (must match train_sam_standalone.py) ────────────────────

class SAMActorCriticEval(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512,512),
                 n_stocks=30, n_tech=8, n_signals=4):
        super().__init__()
        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks, n_tech=n_tech, n_signals=n_signals,
            d_model=64, n_heads=4,
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

        def mlp(sizes):
            ls = []
            for j in range(len(sizes)-1):
                a = nn.ReLU if j < len(sizes)-2 else nn.Identity
                ls += [nn.Linear(sizes[j], sizes[j+1]), a()]
            return nn.Sequential(*ls)

        self.pi_net = mlp([obs_dim] + list(hidden) + [act_dim])
        self.v_net  = mlp([obs_dim] + list(hidden) + [1])

    def act(self, obs):
        with torch.no_grad():
            enc = self.encoder(obs)
            mu  = self.pi_net(enc)
        return mu.numpy()


def load_sam_model(path):
    ckpt    = torch.load(path, map_location="cpu")
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    n_stocks = ckpt.get("n_stocks", STOCK_DIM)
    n_tech   = ckpt.get("n_tech", len(INDICATORS))
    n_sig    = ckpt.get("n_signals", 4)
    model   = SAMActorCriticEval(obs_dim, act_dim, n_stocks=n_stocks,
                                  n_tech=n_tech, n_signals=n_sig)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─── Backtest runner ─────────────────────────────────────────────────────────

def run_model(model, trade_df, n):
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
    return pvs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cppo_model", default="trained_models/agent_cppo_multi_signal_30_epochs.pth")
    parser.add_argument("--sam_standalone", default="trained_models/agent_sam_ppo_30e.pth")
    parser.add_argument("--sam_mpi", default="trained_models/agent_cppo_sam_30_epochs.pth")
    args = parser.parse_args()

    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("paper/figures",    exist_ok=True)

    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    n = len(unique_dates)

    bh = (INITIAL_AMOUNT *
          trade.reset_index().groupby("date")["close"].mean().sort_index().values /
          trade.reset_index().groupby("date")["close"].mean().sort_index().values[0])[:n]

    print(f"Loading CPPO: {args.cppo_model}")
    cppo = load_cppo_model(args.cppo_model, obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)

    rows_metrics = []
    pvs_cppo = None
    pvs_std = None
    pvs_mpi = None

    print("Running CPPO …")
    pvs_cppo = run_model(cppo, trade, n)
    m_cppo = compute_full_metrics(pvs_cppo, bh, name="CPPO (MLP)")
    rows_metrics.append(m_cppo)
    print_metrics(m_cppo)

    if os.path.exists(args.sam_standalone):
        print(f"Loading SAM standalone: {args.sam_standalone}")
        sam_std = load_sam_model(args.sam_standalone)
        print("Running SAM standalone …")
        pvs_std = run_model(sam_std, trade, n)
        m_std = compute_full_metrics(pvs_std, bh, name="SAM-PPO (standalone)")
        rows_metrics.append(m_std)
        print_metrics(m_std)
    else:
        print(f"Skip standalone SAM (missing): {args.sam_standalone}")

    if os.path.exists(args.sam_mpi):
        print(f"Loading SAM-MPI (train_cppo_sam): {args.sam_mpi}")
        sam_mpi = load_sam_mpi_cppo(
            args.sam_mpi,
            obs_dim=STATE_SPACE,
            act_dim=ACTION_SPACE,
            n_stocks=STOCK_DIM,
            n_tech=len(INDICATORS),
            n_signals=4,
        )
        print("Running SAM-MPI …")
        pvs_mpi = run_model(sam_mpi, trade, n)
        m_mpi = compute_full_metrics(pvs_mpi, bh, name="SAM-CPPO (MPI)")
        rows_metrics.append(m_mpi)
        print_metrics(m_mpi)
    else:
        print(f"Skip SAM-MPI (missing): {args.sam_mpi}")

    m_bh = compute_full_metrics(bh, bh, name="Buy & Hold (EW)")
    rows_metrics.append(m_bh)

    print(f"\n{'═'*60}")
    print("  Architecture comparison (CPPO baseline)")
    print(f"{'═'*60}")
    base_cr = float(m_cppo["cumulative_return"])
    base_sr = float(m_cppo["sharpe_ratio"])
    for r in rows_metrics[:-1]:
        if r["name"] == "CPPO (MLP)":
            continue
        try:
            d_cr = float(r["cumulative_return"]) - base_cr
            d_sr = float(r["sharpe_ratio"]) - base_sr
            print(f"  {r['name']:<26}: ΔCR={d_cr:+.2f} pp   ΔSharpe={d_sr:+.4f}")
        except Exception:
            print(f"  {r['name']:<26}: (metrics unavailable)")

    pd.DataFrame(rows_metrics).to_csv(
        "backtest_results/sam_comparison.csv", index=False)
    print("\nSaved → backtest_results/sam_comparison.csv")

    if pvs_std is not None and pvs_mpi is not None:
        d_pv = float(np.max(np.abs(np.array(pvs_std) - np.array(pvs_mpi))))
        if d_pv < 1e-3:
            print("\n  NOTE: SAM standalone and SAM-MPI yield identical portfolios.")
            print("  Checkpoints share identical pi_net layers (tensor-equal); only encoder weights differ.")
            print("  On this dataset, both policies produce the same deterministic mean actions.")

    dates_dt = pd.to_datetime(unique_dates[:n])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3,1]})

    ax1.plot(dates_dt, np.array(pvs_cppo)/pvs_cppo[0],
             color="#2563EB", lw=2.0,
             label=f"CPPO (MLP)  Sharpe={m_cppo['sharpe_ratio']:.2f}")
    if pvs_std is not None:
        m_std = next(r for r in rows_metrics if r["name"] == "SAM-PPO (standalone)")
        ax1.plot(dates_dt, np.array(pvs_std)/pvs_std[0],
                 color="#10B981", lw=2.0, ls="--",
                 label=f"SAM standalone  Sharpe={m_std['sharpe_ratio']:.2f}")
    if pvs_mpi is not None:
        m_mpi = next(r for r in rows_metrics if r["name"] == "SAM-CPPO (MPI)")
        ax1.plot(dates_dt, np.array(pvs_mpi)/pvs_mpi[0],
                 color="#F59E0B", lw=2.0, ls="-.",
                 label=f"SAM-CPPO (MPI)  Sharpe={m_mpi['sharpe_ratio']:.2f}")
    ax1.plot(dates_dt, bh[:n]/bh[0],
             color="#94A3B8", lw=1.5, ls=":", label="Buy & Hold (EW)")
    ax1.set_title("Architecture Ablation: SAM (cross-attention) vs CPPO (plain MLP)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Normalised Portfolio Value")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Drawdown comparison
    dd_specs = [(pvs_cppo, "#2563EB", "CPPO")]
    if pvs_std is not None:
        dd_specs.append((pvs_std, "#10B981", "SAM std"))
    if pvs_mpi is not None:
        dd_specs.append((pvs_mpi, "#F59E0B", "SAM MPI"))
    for pvs, col, lbl in dd_specs:
        pv = np.array(pvs[:n])
        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak * 100
        ax2.fill_between(dates_dt, dd, 0, alpha=0.25, color=col, label=lbl)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig("paper/figures/fig_sam_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved → paper/figures/fig_sam_comparison.png")


if __name__ == "__main__":
    main()
