"""
CPPO training with 4-signal LLM state and drawdown-penalized reward.

Extends train_cppo_llm_risk.py with:
  - 4 LLM signal dimensions per stock (sentiment, risk, confidence, volatility_forecast)
  - Drawdown penalty baked into the environment reward (via env_stocktrading_multi_signal)
  - Slightly larger hidden layers to handle the wider state

Run:
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np 8 python train_cppo_multi_signal.py

Or for a quick local test (4 CPUs, fewer epochs):
    mpirun -np 4 python train_cppo_multi_signal.py --epochs 20 --cpu 4
"""

import argparse
import time
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from gymnasium.spaces import Box, Discrete

from datasets import load_dataset
import pandas as pd

from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from env_stocktrading_multi_signal import StockTradingEnv

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from spinup.utils.run_utils import setup_logger_kwargs

check_and_make_directories([TRAINED_MODEL_DIR])

# Number of LLM signal dimensions added per stock
LLM_DIM = 4  # sentiment, risk, confidence, volatility_forecast

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_train_data(hf_dataset: str = "benstaf/nasdaq_2013_2023",
                    data_file: str = "train_data_multi_signal_2013_2018.csv",
                    local_file: str | None = None) -> pd.DataFrame:
    if local_file:
        train = pd.read_csv(local_file)
    else:
        dataset = load_dataset(hf_dataset, data_files=data_file)
        train = pd.DataFrame(dataset["train"])

    if "Unnamed: 0" in train.columns:
        train = train.drop("Unnamed: 0", axis=1)

    unique_dates = train["date"].unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    # Fill missing LLM signals with neutral values
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


# --------------------------------------------------------------------------- #
# Actor-Critic network
# --------------------------------------------------------------------------- #

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum(np.prod(p.shape) for p in module.parameters())


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(512, 512),
        activation=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            from torch.distributions.categorical import Categorical

            class MLPCategoricalActor(Actor):
                def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
                    super().__init__()
                    self.logits_net = mlp(
                        [obs_dim] + list(hidden_sizes) + [act_dim], activation
                    )

                def _distribution(self, obs):
                    return Categorical(logits=self.logits_net(obs))

                def _log_prob_from_distribution(self, pi, act):
                    return pi.log_prob(act)

            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #

class CPPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.valupdate_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, valupdate, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew.item() if hasattr(rew, "item") else float(rew)
        self.val_buf[self.ptr] = val.item() if hasattr(val, "item") else float(val)
        self.valupdate_buf[self.ptr] = (
            valupdate.item() if hasattr(valupdate, "item") else float(valupdate)
        )
        self.logp_buf[self.ptr] = logp.item() if hasattr(logp, "item") else float(logp)
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf = self.adv_buf - self.valupdate_buf
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# --------------------------------------------------------------------------- #
# CPPO training loop
# --------------------------------------------------------------------------- #

def cppo(
    env_fn,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[512, 512], activation=torch.nn.ReLU),
    seed=42,
    steps_per_epoch=20000,
    epochs=100,
    gamma=0.995,
    clip_ratio=0.7,
    pi_lr=3e-5,
    vf_lr=1e-4,
    train_pi_iters=100,
    train_v_iters=100,
    lam=0.95,
    max_ep_len=3000,
    target_kl=0.35,
    logger_kwargs=dict(),
    save_freq=10,
    alpha=0.85,
    beta=3000.0,
    nu_lr=5e-4,
    lam_lr=5e-4,
    nu_start=0.1,
    lam_start=0.01,
    nu_delay=0.75,
    lam_low_bound=0.001,
    delay=1.0,
    cvar_clip_ratio=0.05,
):
    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log(f"\nNumber of parameters: \t pi: {var_counts[0]}, \t v: {var_counts[1]}\n")

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    nu = nu_start
    cvarlam = lam_start

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clipfrac)

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log(f"Early stopping at step {i} due to reaching max kl.")
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )

    start_time = time.time()
    _reset_out = env.reset()
    o = _reset_out[0] if isinstance(_reset_out, tuple) else _reset_out
    o = np.array(o, dtype=np.float32)
    ep_ret, ep_len = 0, 0

    stock_dimension = env.stock_dim

    for epoch in range(epochs):
        trajectory_num = 0
        bad_trajectory_num = 0
        cvarlam = cvarlam + lam_lr * (beta - nu)
        lam_delta = 0
        nu_delta = 0
        update_num = 0

        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            step_out = env.step(a)
            # gymnasium returns (obs, reward, terminated, truncated, info)
            # gym returns       (obs, reward, done, info)
            if len(step_out) == 5:
                next_o, r, terminated, truncated, _ = step_out
                d = terminated or truncated
            else:
                next_o, r, d, _ = step_out
            next_o = np.array(next_o, dtype=np.float32)
            ep_ret += r
            ep_len += 1

            # Extract LLM risk scores from next observation for CPPO constraint
            llm_risks = next_o[-stock_dimension:]

            risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
            llm_risks_int = np.clip(np.round(llm_risks).astype(int), 1, 5)
            llm_risks_weights = np.array([risk_to_weight.get(int(r), 1.0) for r in llm_risks_int])

            prices = next_o[1 : stock_dimension + 1]
            shares = next_o[stock_dimension + 1 : stock_dimension * 2 + 1]
            stock_values = prices * shares
            total_value = np.sum(stock_values)

            if total_value == 0:
                llm_risk_factor = 1
            else:
                stock_weights = stock_values / total_value
                llm_risk_factor = np.dot(stock_weights, llm_risks_weights)

            adjusted_D_pi = llm_risk_factor * (ep_ret + v - r)
            trajectory_num += 1
            nu_delta += adjusted_D_pi
            updates = np.float32(0.0)

            if adjusted_D_pi < nu:
                bad_trajectory_num += 1
                lam_delta += adjusted_D_pi
                updates = delay * cvarlam / (1 - alpha) * (nu - adjusted_D_pi)
                if updates > abs(v) * cvar_clip_ratio:
                    updates = abs(v) * cvar_clip_ratio
                    update_num += 1
                updates = np.float32(updates)

            buf.store(o, a, r, v, updates, logp)
            logger.store(VVals=v)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(f"Warning: trajectory cut off by epoch at {ep_len} steps.", flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                _reset_out = env.reset()
                o = _reset_out[0] if isinstance(_reset_out, tuple) else _reset_out
                o = np.array(o, dtype=np.float32)
                ep_ret, ep_len = 0, 0

        if bad_trajectory_num > 0:
            lam_delta = lam_delta / bad_trajectory_num
        if trajectory_num > 0:
            nu_delta = nu_delta / trajectory_num
        nu = nu_delta * nu_delay

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        update()

        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

        print("-" * 37)
        print(f"bad_trajectory_num: {bad_trajectory_num}")
        print(f"update_num: {update_num}")
        print(f"nu: {nu}")
        print(f"lam: {cvarlam}")
        print("-" * 37, flush=True)

    return ac


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="cppo_multi_signal")
    parser.add_argument("--local_data", type=str, default=None,
                        help="Path to local train CSV (skips HuggingFace download)")
    parser.add_argument("--drawdown_penalty", type=float, default=0.1)
    # Absorb Jupyter / MPI pass-through args
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ---- Load data ----
    train = load_train_data(local_file=args.local_data)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + (LLM_DIM + len(INDICATORS)) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "drawdown_penalty": args.drawdown_penalty,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    # DummyVecEnv not needed — spinup CPPO calls the env factory directly

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trained_ac = cppo(
        lambda: args.env if hasattr(args, "env") else e_train_gym,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=[args.hid] * args.l,
            activation=torch.nn.ReLU,
        ),
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )

    model_path = (
        TRAINED_MODEL_DIR
        + f"/agent_cppo_multi_signal_{args.epochs}_epochs.pth"
    )
    torch.save(trained_ac.state_dict(), model_path)
    print(f"Training finished — model saved to {model_path}")


if __name__ == "__main__":
    main()
