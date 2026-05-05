"""
SAC (Soft Actor-Critic) baseline with same 4 LLM signals.

Why SAC? It answers the reviewer question:
  "Is the performance from CPPO or from the LLM signals?"

By running SAC with identical signals, any performance gap between
CPPO and SAC is attributable to the RL algorithm, while the gap
between either RL agent and the no-signal baseline is attributable
to the LLM signals.

SAC is off-policy (uses replay buffer) vs CPPO's on-policy PPO.
SAC is generally more sample-efficient but less stable.

Usage:
    python train_sac_baseline.py --local_data train_data_multi_signal_2013_2018.csv
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
from torch.optim import Adam
from collections import deque
import random

try:
    from finrl.config import INDICATORS, TRAINED_MODEL_DIR
except Exception:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
    TRAINED_MODEL_DIR = "trained_models"

os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

from env_stocktrading_multi_signal import StockTradingEnv

LLM_DIM   = 4
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON   = 1e-6


# ─── Network definitions ──────────────────────────────────────────────────────

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SACGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net     = mlp([obs_dim] + list(hidden))
        self.mu_fc   = nn.Linear(hidden[-1], act_dim)
        self.log_std_fc = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        x       = self.net(obs)
        mu      = self.mu_fc(x)
        log_std = self.log_std_fc(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        x_t     = dist.rsample()
        y_t     = torch.tanh(x_t)
        action  = y_t
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)

    def act(self, obs):
        with torch.no_grad():
            a, _, _ = self(obs)
        return a.numpy()


class SACQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim] + list(hidden) + [1])
        self.q2 = mlp([obs_dim + act_dim] + list(hidden) + [1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# ─── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(act),
            torch.FloatTensor(rew).unsqueeze(1),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


# ─── SAC agent ────────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden=(256,256),
                 lr=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, auto_alpha=True, batch_size=256):
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        self.policy   = SACGaussianPolicy(obs_dim, act_dim, hidden)
        self.critic   = SACQNetwork(obs_dim, act_dim, hidden)
        self.critic_t = SACQNetwork(obs_dim, act_dim, hidden)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.policy_opt = Adam(self.policy.parameters(), lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.target_entropy = -act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_opt = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer()

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        return self.policy.act(obs_t)[0]

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_act, next_log_pi, _ = self.policy(next_obs)
            q1_t, q2_t = self.critic_t(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            q_target = rew + self.gamma * (1 - done) * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        act_new, log_pi, _ = self.policy(obs)
        q1_new, q2_new = self.critic(obs, act_new)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_pi - q_new).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(local_file):
    df = pd.read_csv(local_file)
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


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    s = int(args.seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    train_df  = load_data(args.local_data)
    stock_dim = len(train_df.tic.unique())
    state_space = 1 + 2*stock_dim + (LLM_DIM + len(INDICATORS)) * stock_dim

    env_kwargs = dict(
        hmax=100, initial_amount=1_000_000,
        num_stock_shares=[0]*stock_dim,
        buy_cost_pct=[0.001]*stock_dim, sell_cost_pct=[0.001]*stock_dim,
        state_space=state_space, stock_dim=stock_dim,
        tech_indicator_list=INDICATORS, action_space=stock_dim,
        reward_scaling=1e-4, drawdown_penalty=0.1,
    )
    env = StockTradingEnv(df=train_df, **env_kwargs)

    print(f"SAC Training | obs={state_space} act={stock_dim}")
    agent = SACAgent(state_space, stock_dim,
                     hidden=(256,256), lr=3e-4,
                     batch_size=args.batch_size)

    total_steps = 0
    best_return = -np.inf

    for ep in range(args.episodes):
        reset_out = env.reset()
        obs  = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
        ep_ret = 0.0
        done   = False

        while not done:
            if total_steps < args.warmup:
                act = env.action_space.sample()
            else:
                act = agent.select_action(obs)

            step_out = env.step(act)
            next_obs, rew, terminated, truncated, _ = step_out
            done     = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32)
            ep_ret  += float(rew)

            agent.buffer.push(obs, act, float(rew), next_obs, float(done))
            agent.update()
            obs = next_obs
            total_steps += 1

        if ep_ret > best_return:
            best_return = ep_ret

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{args.episodes}  "
                  f"EpRet={ep_ret:.2f}  Best={best_return:.2f}  "
                  f"Alpha={agent.alpha:.4f}  Buffer={len(agent.buffer)}")

    model_path = f"{TRAINED_MODEL_DIR}/agent_sac_llm_{args.episodes}_ep.pth"
    torch.save({
        "policy":   agent.policy.state_dict(),
        "obs_dim":  state_space,
        "act_dim":  stock_dim,
    }, model_path)
    print(f"\nSAC model saved → {model_path}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_data", default="train_data_multi_signal_2013_2018.csv")
    parser.add_argument("--episodes",   type=int, default=300)
    parser.add_argument("--warmup",     type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=0, help="RNG seed for numpy/torch/random.")
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
