"""
Train a PPO agent for Uniswap v3 Liquidity Provisioning.

This is the second RL environment, demonstrating that:
  1. The CPPO+LLM framework generalises beyond stock trading
  2. LLM volatility forecasts improve LP range decisions
  3. Sentiment/risk signals help time rebalancing

Three conditions compared:
  a) PPO + LLM signals (sent, risk, conf, vol_forecast)
  b) PPO + neutral signals (all 3.0)
  c) Passive LP (always full range, 0.3% fee)

Usage:
    python train_uniswap_lp.py --mode train
    python train_uniswap_lp.py --mode eval   # uses existing model
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from env_uniswap_lp import UniswapLPEnv, make_synthetic_pool, LLM_COLS

os.makedirs("trained_models", exist_ok=True)


# ─── PPO hyper-params ─────────────────────────────────────────────────────────

GAMMA      = 0.99
LAM        = 0.95
CLIP_RATIO = 0.2
TARGET_KL  = 0.01
PI_LR      = 3e-4
VF_LR      = 1e-3
TRAIN_PI   = 50
TRAIN_V    = 50


# ─── Actor-Critic ─────────────────────────────────────────────────────────────

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class LPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256,256)):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.pi_net  = mlp([obs_dim] + list(hidden) + [act_dim])
        self.v_net   = mlp([obs_dim] + list(hidden) + [1])

    def _dist(self, obs):
        mu  = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def step(self, obs):
        with torch.no_grad():
            dist   = self._dist(obs)
            a      = dist.sample()
            logp_a = dist.log_prob(a).sum(-1)
            v      = self.v_net(obs)
        return a.numpy(), v.numpy().squeeze(), logp_a.numpy().squeeze()

    def act(self, obs):
        return self.step(obs)[0]

    def value(self, obs):
        with torch.no_grad():
            return self.v_net(obs).squeeze().numpy()


# ─── PPO Buffer ───────────────────────────────────────────────────────────────

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs   = np.zeros((size, obs_dim), dtype=np.float32)
        self.act   = np.zeros((size, act_dim), dtype=np.float32)
        self.rew   = np.zeros(size, dtype=np.float32)
        self.ret   = np.zeros(size, dtype=np.float32)
        self.adv   = np.zeros(size, dtype=np.float32)
        self.val   = np.zeros(size, dtype=np.float32)
        self.logp  = np.zeros(size, dtype=np.float32)
        self.gamma = gamma; self.lam = lam
        self.ptr   = 0; self.path_start = 0; self.max_size = size

    def store(self, o, a, r, v, lp):
        self.obs[self.ptr]  = o
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.val[self.ptr]  = v
        self.logp[self.ptr] = lp
        self.ptr += 1

    def finish_path(self, last_val=0):
        s = slice(self.path_start, self.ptr)
        r = np.append(self.rew[s], last_val)
        v = np.append(self.val[s], last_val)
        d = r[:-1] + self.gamma*v[1:] - v[:-1]
        a = np.zeros(len(d), dtype=np.float32)
        a[-1] = d[-1]
        for t in reversed(range(len(d)-1)):
            a[t] = d[t] + self.gamma*self.lam*a[t+1]
        self.adv[s] = a
        self.ret[s] = a + v[:-1]
        self.path_start = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0; self.path_start = 0
        a = self.adv; a = (a - a.mean()) / (a.std() + 1e-8)
        self.adv = a
        return {k: torch.as_tensor(v) for k, v in
                zip(["obs","act","adv","ret","logp"],
                    [self.obs, self.act, self.adv, self.ret, self.logp])}


# ─── Training ─────────────────────────────────────────────────────────────────

def train_agent(df: pd.DataFrame, mode: str = "llm",
                epochs: int = 50, steps: int = 251,
                seed: int = 0) -> tuple:
    torch.manual_seed(seed); np.random.seed(seed)

    if mode == "neutral":
        df = df.copy()
        for c in LLM_COLS:
            df[c] = 3.0

    env = UniswapLPEnv(df)
    ac  = LPActorCritic(obs_dim=UniswapLPEnv.OBS_DIM, act_dim=UniswapLPEnv.ACT_DIM)
    pi_opt = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=PI_LR)
    v_opt  = Adam(ac.v_net.parameters(), lr=VF_LR)
    buf    = PPOBuffer(UniswapLPEnv.OBS_DIM, UniswapLPEnv.ACT_DIM, steps, GAMMA, LAM)

    def pi_loss(data):
        obs, act, adv, lp_old = data["obs"], data["act"], data["adv"], data["logp"]
        dist  = ac._dist(obs)
        lp    = dist.log_prob(act).sum(-1)
        ratio = torch.exp(lp - lp_old)
        clip  = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
        loss  = -torch.min(ratio*adv, clip*adv).mean()
        kl    = (lp_old - lp).mean().item()
        return loss, kl

    def v_loss(data):
        return ((ac.v_net(data["obs"]).squeeze() - data["ret"]) ** 2).mean()

    def update():
        d = buf.get()
        for _ in range(TRAIN_PI):
            pi_opt.zero_grad()
            lp, kl = pi_loss(d)
            if kl > 1.5 * TARGET_KL:
                break
            lp.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            pi_opt.step()
        for _ in range(TRAIN_V):
            v_opt.zero_grad()
            v_loss(d).backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            v_opt.step()

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_rets = []

    for epoch in range(epochs):
        for t in range(steps):
            a, v, lp = ac.step(torch.FloatTensor(obs))
            obs2, rew, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += rew; ep_len += 1
            buf.store(obs, a, rew, v, lp)
            obs = obs2

            if done or t == steps - 1:
                lv = 0 if done else ac.value(torch.FloatTensor(obs))
                buf.finish_path(lv)
                if done:
                    ep_rets.append(ep_ret)
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        update()

        if (epoch + 1) % 10 == 0:
            m = np.mean(ep_rets[-5:]) if ep_rets else 0.0
            print(f"  [{mode:7s}] Epoch {epoch+1:2d}/{epochs}  EpRet={m:.4f}")

    return ac, ep_rets


# ─── Evaluation ───────────────────────────────────────────────────────────────

def eval_passive_lp(df: pd.DataFrame) -> list[float]:
    """Baseline: always full range [0, ∞), earns minimal fees, no IL."""
    env = UniswapLPEnv(df)
    obs, _ = env.reset()
    # Always: center=0, full width, full allocation
    action = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    values = [env.initial_cap]
    done   = False
    while not done:
        obs, rew, term, trunc, info = env.step(action)
        done = term or trunc
        values.append(env.lp_capital + env.hold_capital + env.fee_cumul)
    return values


def run_eval(ac: "LPActorCritic", df: pd.DataFrame) -> list[float]:
    env    = UniswapLPEnv(df)
    obs, _ = env.reset()
    values = [env.initial_cap]
    done   = False
    while not done:
        a      = ac.act(torch.FloatTensor(obs))
        obs, _, term, trunc, info = env.step(a)
        done   = term or trunc
        values.append(env.lp_capital + env.hold_capital + env.fee_cumul)
    return values


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["train","eval"], default="train")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--n_days",  type=int, default=504)  # 2 years
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("paper/figures", exist_ok=True)
    os.makedirs("backtest_results", exist_ok=True)

    print(f"Uniswap LP RL — {args.mode} mode")
    df = make_synthetic_pool(n_days=args.n_days, seed=args.seed)
    print(f"  Pool data: {len(df)} days  price=[{df['price'].min():.0f},{df['price'].max():.0f}]")

    if args.mode == "train":
        print("\nTraining PPO + LLM signals …")
        ac_llm, rets_llm = train_agent(df, mode="llm",
                                       epochs=args.epochs, seed=args.seed)

        print("\nTraining PPO + neutral signals …")
        ac_neutral, rets_neutral = train_agent(df, mode="neutral",
                                               epochs=args.epochs, seed=args.seed)

        torch.save({"state_dict": ac_llm.state_dict()},
                   "trained_models/uniswap_ppo_llm.pth")
        torch.save({"state_dict": ac_neutral.state_dict()},
                   "trained_models/uniswap_ppo_neutral.pth")
        print("\nModels saved.")
    else:
        # Load trained models
        ac_llm     = LPActorCritic(UniswapLPEnv.OBS_DIM, UniswapLPEnv.ACT_DIM)
        ac_neutral = LPActorCritic(UniswapLPEnv.OBS_DIM, UniswapLPEnv.ACT_DIM)
        p_llm = "trained_models/uniswap_ppo_llm.pth"
        p_neu = "trained_models/uniswap_ppo_neutral.pth"
        if not (os.path.exists(p_llm) and os.path.exists(p_neu)):
            print("Models not found. Run with --mode train first.")
            return
        ac_llm.load_state_dict(torch.load(p_llm)["state_dict"])
        ac_neutral.load_state_dict(torch.load(p_neu)["state_dict"])

    # Evaluation
    print("\nRunning evaluation …")
    df_test  = make_synthetic_pool(n_days=args.n_days, seed=args.seed + 100)
    pvs_llm  = run_eval(ac_llm,     df_test)
    pvs_neu  = run_eval(ac_neutral, df_test)
    pvs_pass = eval_passive_lp(df_test)
    n = min(len(pvs_llm), len(pvs_neu), len(pvs_pass))

    # Metrics
    from metrics_extended import compute_full_metrics
    bh = np.array(pvs_pass[:n])
    m_llm = compute_full_metrics(pvs_llm[:n], bh, name="PPO+LLM", initial_capital=100_000)
    m_neu = compute_full_metrics(pvs_neu[:n], bh, name="PPO+Neutral", initial_capital=100_000)

    print(f"\n  PPO+LLM    : CR={m_llm['cumulative_return']:.2f}%  "
          f"Sharpe={m_llm['sharpe_ratio']:.4f}  MDD={m_llm['max_drawdown_pct']:.2f}%")
    print(f"  PPO+Neutral: CR={m_neu['cumulative_return']:.2f}%  "
          f"Sharpe={m_neu['sharpe_ratio']:.4f}  MDD={m_neu['max_drawdown_pct']:.2f}%")
    print(f"  Passive LP : CR={(pvs_pass[-1]/pvs_pass[0]-1)*100:.2f}%")
    print(f"  Signal ΔSharpe: {m_llm['sharpe_ratio']-m_neu['sharpe_ratio']:+.4f}")

    pd.DataFrame([m_llm, m_neu]).to_csv(
        "backtest_results/uniswap_comparison.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(n)
    ax.plot(x, np.array(pvs_llm[:n])/pvs_llm[0], "#2563EB", lw=2.2,
            label=f"PPO + LLM Signals  (Sharpe {m_llm['sharpe_ratio']:.2f})")
    ax.plot(x, np.array(pvs_neu[:n])/pvs_neu[0], "#F59E0B", lw=1.8, ls="--",
            label=f"PPO + Neutral  (Sharpe {m_neu['sharpe_ratio']:.2f})")
    ax.plot(x, np.array(pvs_pass[:n])/pvs_pass[0], "#94A3B8", lw=1.5, ls=":",
            label="Passive LP (full range)")
    ax.set_title("Uniswap v3 LP: LLM-informed vs Neutral vs Passive",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Days"); ax.set_ylabel("Normalised Portfolio Value")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_uniswap_lp.png", dpi=150)
    plt.close(fig)
    print("Saved → paper/figures/fig_uniswap_lp.png")


if __name__ == "__main__":
    main()
