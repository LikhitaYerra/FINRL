"""
CPPO training with Signal Attention Module (SAM).

Identical to train_cppo_multi_signal.py but uses SAMActorCritic instead of
MLPActorCritic. The SAM pre-processes the observation via cross-attention
before the policy and value networks.

Key architectural difference:
  MLPActorCritic : obs → [concat] → MLP → action
  SAMActorCritic : obs → CrossAttention(tech, signals) → MLP → action

Run:
    mpirun -np 4 python train_cppo_sam.py --local_data train_data_multi_signal_2013_2018.csv
    mpirun -np 4 python train_cppo_sam.py --local_data train_data_multi_signal_2013_2018.csv --epochs 30
"""

import argparse
import time
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
from gymnasium.spaces import Box

import pandas as pd

# Import only what we need from finrl to avoid the broken __init__.py chain
try:
    from finrl.config import INDICATORS, TRAINED_MODEL_DIR
except Exception:
    INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
    TRAINED_MODEL_DIR = "trained_models"

import os as _os
_os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

from env_stocktrading_multi_signal import StockTradingEnv
from signal_attention import SAMActorCritic, SAMStateEncoder

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import (
    mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs,
)
from spinup.utils.run_utils import setup_logger_kwargs

LLM_DIM = 4


# ─── Data loading ────────────────────────────────────────────────────────────

def load_train_data(local_file=None):
    if local_file:
        train = pd.read_csv(local_file)
    else:
        from datasets import load_dataset
        dataset = load_dataset("benstaf/nasdaq_2013_2023",
                               data_files="train_data_multi_signal_2013_2018.csv")
        train = pd.DataFrame(dataset["train"])

    if "Unnamed: 0" in train.columns:
        train = train.drop("Unnamed: 0", axis=1)

    unique_dates = train["date"].unique()
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    for col, default in [
        ("llm_sentiment", 3),
        ("llm_risk", 3),
        ("llm_confidence", 3),
        ("llm_volatility_forecast", 3),
    ]:
        if col not in train.columns:
            train[col] = default
        else:
            train[col].fillna(default, inplace=True)

    return train


# ─── SAM Actor (wraps SAMActorCritic as a SpinningUp-compatible actor) ────────

class SAMActor(nn.Module):
    """SpinningUp-style Actor that pre-processes obs via SAM."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
                 n_stocks, n_tech, n_signals, sam_d_model, sam_n_heads):
        super().__init__()
        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks, n_tech=n_tech, n_signals=n_signals,
            d_model=sam_d_model, n_heads=sam_n_heads,
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        # MLP after encoding
        layers = []
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.mu_net = nn.Sequential(*layers)

    def _distribution(self, obs):
        enc = self.encoder(obs)
        mu  = self.mu_net(enc)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi     = self._distribution(obs)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, logp_a


class SAMCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation,
                 n_stocks, n_tech, n_signals, sam_d_model, sam_n_heads):
        super().__init__()
        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks, n_tech=n_tech, n_signals=n_signals,
            d_model=sam_d_model, n_heads=sam_n_heads,
        )
        layers = []
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.v_net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(self.encoder(obs)), -1)


class SAMActorCriticTraining(nn.Module):
    """
    Full SAM Actor-Critic compatible with SpinningUp CPPO.
    Uses a SINGLE shared SAMStateEncoder for both pi and v
    to keep parameter count within MPI buffer limits.
    """

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(512, 512), activation=nn.ReLU,
                 n_stocks=30, n_tech=8, n_signals=4,
                 sam_d_model=64, sam_n_heads=4):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Single shared encoder — avoids doubling params and MPI buffer overflow
        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks, n_tech=n_tech, n_signals=n_signals,
            d_model=sam_d_model, n_heads=sam_n_heads,
        )

        # Policy head
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        layers = []
        for j in range(len(sizes)-1):
            a = activation if j < len(sizes)-2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j+1]), a()]
        self.pi_net = nn.Sequential(*layers)

        # Value head
        v_sizes = [obs_dim] + list(hidden_sizes) + [1]
        v_layers = []
        for j in range(len(v_sizes)-1):
            a = activation if j < len(v_sizes)-2 else nn.Identity
            v_layers += [nn.Linear(v_sizes[j], v_sizes[j+1]), a()]
        self.v_net = nn.Sequential(*v_layers)

    def _encode(self, obs):
        return self.encoder(obs)

    def _distribution(self, obs):
        enc = self._encode(obs)
        mu  = self.pi_net(enc)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def step(self, obs):
        with torch.no_grad():
            enc    = self._encode(obs)
            mu     = self.pi_net(enc)
            std    = torch.exp(self.log_std)
            dist   = Normal(mu, std)
            a      = dist.sample()
            logp_a = dist.log_prob(a).sum(axis=-1)
            v      = self.v_net(enc)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# ─── PPO Buffer ───────────────────────────────────────────────────────────────
# (copied verbatim from train_cppo_multi_signal.py)

def combined_shape(length, shape=None):
    if shape is None: return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim),  dtype=np.float32)
        self.act_buf  = np.zeros(combined_shape(size, act_dim),  dtype=np.float32)
        self.adv_buf  = np.zeros(size, dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.ret_buf  = np.zeros(size, dtype=np.float32)
        self.val_buf  = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-9)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# ─── CPPO training loop ───────────────────────────────────────────────────────

def cppo(env_fn, actor_critic=SAMActorCriticTraining, ac_kwargs=None,
         seed=0, steps_per_epoch=4096, epochs=30, gamma=0.99, clip_ratio=0.2,
         pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
         lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=None,
         cvar_alpha=0.05, cvar_weight=0.1):
    if ac_kwargs is None:
        ac_kwargs = {}
    if logger_kwargs is None:
        logger_kwargs = {}

    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env       = env_fn()
    obs_dim   = env.observation_space.shape
    act_dim   = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac)

    n_params = sum(p.numel() for p in ac.parameters())
    if proc_id() == 0:
        print(f"  [SAM-CPPO] Parameters: {n_params:,}")

    local_steps = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        dist = ac._distribution(obs)
        logp = ac._log_prob(dist, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()
        # CVaR penalty on negative advantages (tail-risk regularisation)
        neg_adv = adv[adv < 0]
        cvar_pen = 0.0
        if len(neg_adv) > 0:
            k = max(1, int(cvar_alpha * len(neg_adv)))
            cvar_pen = cvar_weight * neg_adv.topk(k, largest=False).values.mean().abs()
        approx_kl = (logp_old - logp).mean().item()
        return loss_pi + cvar_pen, approx_kl

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        enc = ac._encode(obs)
        return ((ac.v_net(enc) - ret) ** 2).mean()

    # Optimizer covers all parameters (shared encoder + heads)
    all_params = list(ac.parameters())
    pi_params  = list(ac.encoder.parameters()) + list(ac.pi_net.parameters()) + [ac.log_std]
    v_params   = list(ac.encoder.parameters()) + list(ac.v_net.parameters())
    pi_opt = Adam(pi_params, lr=pi_lr)
    v_opt  = Adam(v_params,  lr=vf_lr)

    def safe_mpi_avg_grads(module):
        """mpi_avg_grads that skips parameters with no gradient (None grad)."""
        if num_procs() == 1:
            return
        for p in module.parameters():
            if p.grad is None:
                # Initialise to zero so Allreduce can proceed
                p.grad = torch.zeros_like(p.data)
        mpi_avg_grads(module)

    def update():
        data = buf.get()
        for _ in range(train_pi_iters):
            pi_opt.zero_grad(set_to_none=False)
            loss_pi, kl = compute_loss_pi(data)
            if mpi_avg(kl) > 1.5 * target_kl:
                break
            loss_pi.backward()
            safe_mpi_avg_grads(ac)
            pi_opt.step()
        for _ in range(train_v_iters):
            v_opt.zero_grad(set_to_none=False)
            compute_loss_v(data).backward()
            safe_mpi_avg_grads(ac)
            v_opt.step()

    # Training loop
    reset_out = env.reset()
    obs = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(local_steps):
            a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            step_out = env.step(a)
            next_obs, rew, terminated, truncated, _ = step_out
            done = terminated or truncated
            ep_ret += float(rew)
            ep_len += 1
            buf.store(obs, a, float(rew), float(v), float(logp))
            obs = np.array(next_obs, dtype=np.float32)

            timeout = (ep_len == max_ep_len)
            terminal = done or timeout
            epoch_ended = (t == local_steps - 1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                reset_out = env.reset()
                obs = np.array(reset_out[0] if isinstance(reset_out, tuple) else reset_out, dtype=np.float32)
                ep_ret, ep_len = 0, 0

        update()

        if proc_id() == 0 and (epoch + 1) % 5 == 0:
            stats = logger.get_stats("EpRet")
            mean_ret = stats[0] if isinstance(stats, tuple) else stats
            print(f"  Epoch {epoch+1}/{epochs}  EpRet={mean_ret:.2f}")

    return ac


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hid",          type=int,   default=512)
    parser.add_argument("--l",            type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--cpu",          type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--exp_name",     type=str,   default="cppo_sam")
    parser.add_argument("--local_data",   type=str,   default=None)
    parser.add_argument("--sam_d_model",  type=int,   default=64)
    parser.add_argument("--sam_n_heads",  type=int,   default=4)
    parser.add_argument("--cvar_weight",  type=float, default=0.1)
    parser.add_argument("-f", "--file",   type=str)
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    train = load_train_data(local_file=args.local_data)
    stock_dim   = len(train.tic.unique())
    state_space = 1 + 2*stock_dim + (LLM_DIM + len(INDICATORS)) * stock_dim

    env_kwargs = dict(
        hmax=100, initial_amount=1_000_000,
        num_stock_shares=[0]*stock_dim,
        buy_cost_pct=[0.001]*stock_dim, sell_cost_pct=[0.001]*stock_dim,
        state_space=state_space, stock_dim=stock_dim,
        tech_indicator_list=INDICATORS, action_space=stock_dim,
        reward_scaling=1e-4, drawdown_penalty=0.1,
    )

    env_fn = lambda: StockTradingEnv(df=train, **env_kwargs)

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trained_ac = cppo(
        env_fn,
        actor_critic=SAMActorCriticTraining,
        ac_kwargs=dict(
            hidden_sizes=[args.hid] * args.l,
            activation=torch.nn.ReLU,
            n_stocks=stock_dim,
            n_tech=len(INDICATORS),
            n_signals=LLM_DIM,
            sam_d_model=args.sam_d_model,
            sam_n_heads=args.sam_n_heads,
        ),
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        cvar_weight=args.cvar_weight,
    )

    model_path = f"{TRAINED_MODEL_DIR}/agent_cppo_sam_{args.epochs}_epochs.pth"
    torch.save(trained_ac.state_dict(), model_path)
    if proc_id() == 0:
        print(f"SAM model saved → {model_path}")


if __name__ == "__main__":
    main()
