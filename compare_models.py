"""
Compare baseline (neutral signals) vs real-signal CPPO model performance.

Usage:
    python compare_models.py
"""

import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from env_stocktrading_multi_signal import StockTradingEnv
from finrl.config import INDICATORS


# ── Actor-Critic (same architecture as training) ─────────────────────────────

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.log_std = nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)))
        self.mu_net  = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    def forward(self, obs):
        return self.mu_net(obs), torch.exp(self.log_std)

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512,512), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
    def act(self, obs):
        with torch.no_grad():
            mu, _ = self.pi(obs)
        return mu.numpy()


# ── Config ────────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_model(model_path, trade_df, unique_dates):
    env = StockTradingEnv(df=trade_df, stock_dim=STOCK_DIM, **ENV_KWARGS)
    sd  = torch.load(model_path, map_location="cpu")
    ac  = MLPActorCritic(obs_dim=STATE_SPACE, act_dim=ACTION_SPACE)
    ac.load_state_dict(sd)
    ac.eval()

    reset_out = env.reset()
    obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    done = False
    pvs  = [INITIAL_AMOUNT]

    while not done:
        action = ac.act(torch.FloatTensor(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        obs  = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        pvs.append(float(env.asset_memory[-1]) if env.asset_memory else pvs[-1])

    return pvs[:len(unique_dates)]


def metrics(pvs, name=""):
    pv  = np.array(pvs, dtype=float)
    ret = np.diff(pv) / pv[:-1]
    cum = (pv[-1] / pv[0] - 1) * 100
    ann = ((pv[-1] / pv[0]) ** (252 / len(pv)) - 1) * 100
    sh  = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252)
    rm  = np.maximum.accumulate(pv)
    mdd = ((pv - rm) / rm).min() * 100
    neg = ret[ret < 0].std() + 1e-9
    so  = (ret.mean() / neg) * np.sqrt(252)
    cal = ann / abs(mdd + 1e-9)
    return dict(name=name, cum=round(cum,2), ann=round(ann,2),
                sharpe=round(sh,4), sortino=round(so,4),
                mdd=round(mdd,2), calmar=round(cal,4))


def bh_series(trade_df):
    avg = trade_df.groupby("date")["close"].mean().sort_index()
    return (INITIAL_AMOUNT * avg.values / avg.values[0]).tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("backtest_results", exist_ok=True)

    print("Loading trade data …")
    trade = pd.read_csv("trade_data_multi_signal_2019_2023.csv")
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")
    for col in ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]:
        trade[col] = trade.get(col, 3.0).fillna(3.0)

    models = {
        "CPPO (neutral signals)": "trained_models/agent_cppo_baseline_neutral.pth",
        "CPPO (real LLM signals)": "trained_models/agent_cppo_multi_signal_30_epochs.pth",
    }
    # Use new model if available
    if os.path.exists("trained_models/agent_cppo_multi_signal_30_epochs.pth"):
        models["CPPO (real LLM signals)"] = "trained_models/agent_cppo_multi_signal_30_epochs.pth"

    results = {}
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"  Skipping {name} — model not found")
            continue
        print(f"Running {name} …")
        pvs = run_model(path, trade, unique_dates)
        results[name] = pvs
        m = metrics(pvs, name)
        print(f"  Cum={m['cum']}%  Sharpe={m['sharpe']}  MDD={m['mdd']}%  Sortino={m['sortino']}")

    bh = bh_series(trade)
    results["Buy & Hold (EW)"] = bh[:len(unique_dates)]

    # ── Plot ─────────────────────────────────────────────────────────────────
    dates_dt = pd.to_datetime(unique_dates)
    colors   = {"CPPO (neutral signals)": "#94A3B8",
                "CPPO (real LLM signals)": "#2563EB",
                "Buy & Hold (EW)": "#F59E0B"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for name, pvs in results.items():
        norm = [v / pvs[0] for v in pvs]
        axes[0].plot(dates_dt[:len(pvs)], norm,
                     label=name, color=colors.get(name, "#888"), linewidth=1.8, alpha=0.9)

    axes[0].set_title("Portfolio Value — Model Comparison (2019–2023)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Normalised Value (start=1)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for name, pvs in results.items():
        pv  = np.array(pvs, dtype=float)
        rm  = np.maximum.accumulate(pv)
        dd  = (pv - rm) / rm * 100
        axes[1].plot(dates_dt[:len(pvs)], dd,
                     label=name, color=colors.get(name, "#888"), linewidth=1.4, alpha=0.85)

    axes[1].set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    path = "backtest_results/model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for name, pvs in results.items():
        rows.append(metrics(pvs, name))
    summary = pd.DataFrame(rows).set_index("name")
    print("\n── Comparison Summary ─────────────────────────────────────")
    print(summary.to_string())
    summary.to_csv("backtest_results/model_comparison.csv")


if __name__ == "__main__":
    main()
