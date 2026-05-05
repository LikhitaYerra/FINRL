"""
Standalone SAM-PPO Training (no SpinningUp / no MPI)

Clean, self-contained PPO implementation for the SAM Actor-Critic.
Produces a model-compatible with backtest_sam.py.

Usage:
    python train_sam_standalone.py --local_data train_data_multi_signal_2013_2018.csv
    python train_sam_standalone.py --epochs 30 --steps 4000 --seed 0
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# Finrl must come before torch to avoid tensorboard/tensorflow state pollution
try:
    from finrl.config import INDICATORS
except BaseException:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

os.makedirs("trained_models", exist_ok=True)

from env_stocktrading_multi_signal import StockTradingEnv
from signal_attention import SAMStateEncoder


# ─── PPO hyper-params ─────────────────────────────────────────────────────────

GAMMA       = 0.99
LAM         = 0.97
CLIP_RATIO  = 0.2
TARGET_KL   = 0.01
PI_LR       = 3e-4
VF_LR       = 1e-3
TRAIN_PI    = 80
TRAIN_V     = 80


# ─── SAM Actor-Critic (single-process, no MPI) ────────────────────────────────

class SAMActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden=(512, 512),
                 n_stocks=30, n_tech=10, n_signals=4,
                 sam_d=64, sam_heads=4):
        super().__init__()
        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks, n_tech=n_tech, n_signals=n_signals,
            d_model=sam_d, n_heads=sam_heads,
        )

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

        def mlp(sizes, out_act=nn.Identity):
            ls = []
            for j in range(len(sizes)-1):
                a = nn.ReLU if j < len(sizes)-2 else out_act
                ls += [nn.Linear(sizes[j], sizes[j+1]), a()]
            return nn.Sequential(*ls)

        enc_out = obs_dim  # encoder output preserves obs_dim + SAM attention
        self.pi_net = mlp([enc_out] + list(hidden) + [act_dim])
        self.v_net  = mlp([enc_out] + list(hidden) + [1])

    def _encode(self, obs):
        return self.encoder(obs)

    def _pi_dist(self, obs):
        enc = self._encode(obs)
        mu  = self.pi_net(enc)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def step(self, obs):
        with torch.no_grad():
            dist   = self._pi_dist(obs)
            a      = dist.sample()
            logp_a = dist.log_prob(a).sum(axis=-1)
            enc    = self._encode(obs)
            v      = self.v_net(enc)
        return a.numpy(), v.numpy().squeeze(), logp_a.numpy().squeeze()

    def act(self, obs):
        return self.step(obs)[0]

    def value(self, obs):
        with torch.no_grad():
            enc = self._encode(obs)
            return self.v_net(enc).squeeze().numpy()


# ─── PPO buffer ───────────────────────────────────────────────────────────────

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf  = np.zeros(size, dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.ret_buf  = np.zeros(size, dtype=np.float32)
        self.val_buf  = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr = 0; self.path_start_idx = 0; self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.val_buf[self.ptr]  = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = np.zeros(len(deltas), dtype=np.float32)
        adv[-1] = deltas[-1]
        for t in reversed(range(len(deltas)-1)):
            adv[t] = deltas[t] + self.gamma * self.lam * adv[t+1]
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + vals[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0; self.path_start_idx = 0
        adv_mean = self.adv_buf.mean()
        adv_std  = self.adv_buf.std() + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return {k: torch.as_tensor(v) for k, v in zip(
            ["obs","act","adv","ret","logp"],
            [self.obs_buf, self.act_buf, self.adv_buf,
             self.ret_buf, self.logp_buf]
        )}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    unique_dates = df["date"].unique()
    df["new_idx"] = df["date"].map({d: i for i, d in enumerate(unique_dates)})
    df = df.set_index("new_idx")
    for col, default in [("llm_sentiment",3),("llm_risk",3),
                          ("llm_confidence",3),("llm_volatility_forecast",3)]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df  = load_data(args.local_data)
    stock_dim = len(train_df.tic.unique())
    state_dim = 1 + 2*stock_dim + (4 + len(INDICATORS)) * stock_dim
    act_dim   = stock_dim

    env_kwargs = dict(
        hmax=100, initial_amount=1_000_000,
        num_stock_shares=[0]*stock_dim,
        buy_cost_pct=[0.001]*stock_dim, sell_cost_pct=[0.001]*stock_dim,
        state_space=state_dim, stock_dim=stock_dim,
        tech_indicator_list=INDICATORS, action_space=stock_dim,
        reward_scaling=1e-4, drawdown_penalty=0.1,
    )
    env = StockTradingEnv(df=train_df, **env_kwargs)

    n_tech = len(INDICATORS)        # SAMStateEncoder adds +2 (price+holdings) internally
    n_sig  = 4                     # LLM signals per stock

    ac = SAMActorCritic(
        obs_dim=state_dim, act_dim=act_dim,
        n_stocks=stock_dim, n_tech=n_tech, n_signals=n_sig,
    )
    n_params = sum(p.numel() for p in ac.parameters() if p.requires_grad)
    print(f"[SAM-PPO] obs={state_dim}  act={act_dim}  params={n_params:,}")

    pi_opt = Adam(
        list(ac.encoder.parameters()) + list(ac.pi_net.parameters()) + [ac.log_std],
        lr=PI_LR,
    )
    v_opt  = Adam(
        list(ac.encoder.parameters()) + list(ac.v_net.parameters()),
        lr=VF_LR,
    )

    buf = PPOBuffer(state_dim, act_dim, args.steps, GAMMA, LAM)

    def compute_pi_loss(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        dist   = ac._pi_dist(obs)
        logp   = dist.log_prob(act).sum(axis=-1)
        ratio  = torch.exp(logp - logp_old)
        clip_a = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO) * adv
        loss   = -torch.min(ratio * adv, clip_a).mean()
        kl     = (logp_old - logp).mean().item()
        return loss, kl

    def compute_v_loss(data):
        enc = ac._encode(data["obs"])
        return ((ac.v_net(enc).squeeze() - data["ret"]) ** 2).mean()

    def update():
        data = buf.get()
        for _ in range(TRAIN_PI):
            pi_opt.zero_grad()
            loss_pi, kl = compute_pi_loss(data)
            if kl > 1.5 * TARGET_KL:
                break
            loss_pi.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            pi_opt.step()
        for _ in range(TRAIN_V):
            v_opt.zero_grad()
            compute_v_loss(data).backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            v_opt.step()

    reset_out = env.reset()
    obs = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out,
                   dtype=np.float32)
    ep_ret, ep_len = 0.0, 0
    ep_rets = []

    for epoch in range(args.epochs):
        for t in range(args.steps):
            a, v, logp = ac.step(torch.FloatTensor(obs))

            step_out = env.step(a)
            next_obs, rew, terminated, truncated, _ = step_out
            done = terminated or truncated
            ep_ret += float(rew)
            ep_len += 1

            buf.store(obs, a, float(rew), float(v), float(logp))
            obs = np.array(next_obs, dtype=np.float32)

            timeout    = (ep_len == args.max_ep_len)
            terminal   = done or timeout
            epoch_ended = (t == args.steps - 1)

            if terminal or epoch_ended:
                last_val = 0 if terminal else ac.value(torch.FloatTensor(obs))
                buf.finish_path(last_val)
                if terminal:
                    ep_rets.append(ep_ret)
                reset_out = env.reset()
                obs = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out,
                               dtype=np.float32)
                ep_ret, ep_len = 0.0, 0

        update()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            mean_ret = np.mean(ep_rets[-10:]) if ep_rets else 0.0
            print(f"  Epoch {epoch+1:02d}/{args.epochs}  EpRet={mean_ret:,.2f}")

    save_path = f"trained_models/agent_sam_ppo_{args.epochs}e.pth"
    torch.save({
        "state_dict": ac.state_dict(),
        "obs_dim":    state_dim,
        "act_dim":    act_dim,
        "n_stocks":   stock_dim,
        "n_tech":     n_tech,
        "n_signals":  n_sig,
    }, save_path)
    print(f"\nSAM-PPO model saved → {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_data",  default="train_data_multi_signal_2013_2018.csv")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--steps",       type=int,   default=4000)
    parser.add_argument("--max_ep_len",  type=int,   default=1250)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("-f", "--file",  type=str)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
