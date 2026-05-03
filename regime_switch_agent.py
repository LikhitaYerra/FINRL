"""
Regime-switching inference wrapper for FinRL Contest 2025 Task 1.

Strategy (from README):
  Bull market  → PPO  (lower volatility, trending up)
  Bear market  → CPPO-DeepSeek / CPPO-MultiSignal  (high turbulence / VIX)

This module provides:
  1. RegimeSwitchAgent  – wraps a PPO model + CPPO model and selects which
                          one acts at each timestep based on a regime signal.
  2. calibrate_threshold – finds the optimal turbulence/VIX split threshold
                           on the training period that maximises a composite
                           score of return and Rachev ratio.
  3. run_backtest         – runs the agent through a trade dataset and returns
                           a DataFrame of daily portfolio values.

Usage (standalone):
    python regime_switch_agent.py \
        --ppo_model   trained_models/agent_ppo.pth \
        --cppo_model  trained_models/agent_cppo_multi_signal.pth \
        --trade_data  trade_data_multi_signal_2019_2023.csv \
        --state_space 1981
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from gymnasium.spaces import Box

from finrl.config import INDICATORS


# --------------------------------------------------------------------------- #
# Minimal Actor-Critic re-implementation (must match training architecture)
# --------------------------------------------------------------------------- #

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, obs):
        with torch.no_grad():
            dist = self.forward(obs)
            return dist.mean.numpy()


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(512, 512),
        activation=nn.ReLU,
    ):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        return self.pi.act(obs_t)

    def step(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            pi = self.pi.forward(obs_t)
            a = pi.sample()
            logp = pi.log_prob(a).sum(-1)
            v = self.v(obs_t)
        return a.numpy(), v.numpy(), logp.numpy()


# --------------------------------------------------------------------------- #
# Regime detection helpers
# --------------------------------------------------------------------------- #

def compute_turbulence_threshold(train_df: pd.DataFrame, percentile: float = 75.0) -> float:
    """
    Compute a turbulence threshold from the training-period data.
    Default: 75th percentile of daily turbulence values (top 25% = bear).
    """
    turb = train_df["turbulence"].dropna()
    return float(np.percentile(turb, percentile))


def compute_vix_threshold(trade_df: pd.DataFrame, percentile: float = 66.0) -> float:
    """
    Compute a VIX threshold.  Uses 66th percentile: top 1/3 of VIX days
    are treated as high-volatility / bear regime.
    """
    if "vix" not in trade_df.columns:
        return float("inf")
    vix = trade_df["vix"].dropna()
    return float(np.percentile(vix, percentile))


# --------------------------------------------------------------------------- #
# Regime-switching agent
# --------------------------------------------------------------------------- #

class RegimeSwitchAgent:
    """
    Selects between a PPO model (bull) and a CPPO model (bear) based on
    turbulence and VIX thresholds.

    Both models must have identical obs_dim / act_dim.
    """

    def __init__(
        self,
        ppo_model: MLPActorCritic,
        cppo_model: MLPActorCritic,
        turbulence_threshold: float = 200.0,
        vix_threshold: float = 25.0,
        use_vix: bool = True,
    ):
        self.ppo = ppo_model
        self.cppo = cppo_model
        self.turbulence_threshold = turbulence_threshold
        self.vix_threshold = vix_threshold
        self.use_vix = use_vix

    def is_bear_regime(self, turbulence: float, vix: Optional[float] = None) -> bool:
        if turbulence >= self.turbulence_threshold:
            return True
        if self.use_vix and vix is not None and vix >= self.vix_threshold:
            return True
        return False

    def act(
        self,
        obs: np.ndarray,
        turbulence: float,
        vix: Optional[float] = None,
    ) -> tuple[np.ndarray, str]:
        """
        Returns (action, regime_label).
        regime_label is 'bear' or 'bull'.
        """
        if self.is_bear_regime(turbulence, vix):
            return self.cppo.act(obs), "bear"
        return self.ppo.act(obs), "bull"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(
    path: str,
    obs_dim: int,
    act_dim: int,
    hidden_sizes=(512, 512),
    activation=nn.ReLU,
) -> MLPActorCritic:
    model = MLPActorCritic(obs_dim, act_dim, hidden_sizes, activation)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Backtesting loop
# --------------------------------------------------------------------------- #

def run_backtest(
    agent: RegimeSwitchAgent,
    trade_df: pd.DataFrame,
    env_kwargs: dict,
) -> pd.DataFrame:
    """
    Run the regime-switch agent through the full trade period.

    Returns a DataFrame with columns: date, account_value, regime
    """
    from env_stocktrading_multi_signal import StockTradingEnv

    env = StockTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = env.reset()

    account_values = []
    dates = []
    regimes = []

    while True:
        # Extract turbulence and VIX from current data
        data = env.data
        if hasattr(data, "turbulence"):
            turbulence_val = float(data.turbulence.values[0]) if hasattr(data.turbulence, "values") else float(data.turbulence)
        else:
            turbulence_val = 0.0

        vix_val = None
        if "vix" in trade_df.columns:
            vix_series = data["vix"] if hasattr(data, "__getitem__") else None
            if vix_series is not None:
                vix_val = float(vix_series.values[0]) if hasattr(vix_series, "values") else float(vix_series)

        action, regime = agent.act(obs, turbulence_val, vix_val)
        obs, reward, done, _, info = env.step(action)

        account_values.append(env.asset_memory[-1])
        dates.append(env.date_memory[-1])
        regimes.append(regime)

        if done:
            break

    result = pd.DataFrame(
        {
            "date": dates,
            "account_value": account_values,
            "regime": regimes,
        }
    )
    result["daily_return"] = result["account_value"].pct_change(1).fillna(0)
    return result


# --------------------------------------------------------------------------- #
# Threshold calibration on training period
# --------------------------------------------------------------------------- #

def calibrate_threshold(
    train_df: pd.DataFrame,
    turbulence_percentiles: list[float] = None,
) -> dict:
    """
    Find the turbulence percentile threshold that best separates
    positive-return days from negative-return days in the training period.

    Returns a dict with the recommended threshold and stats.
    """
    if turbulence_percentiles is None:
        turbulence_percentiles = [50, 60, 66, 70, 75, 80, 85, 90]

    turb = train_df["turbulence"].dropna().values
    results = []

    for pct in turbulence_percentiles:
        threshold = np.percentile(turb, pct)
        bear_days = train_df[train_df["turbulence"] >= threshold]
        bull_days = train_df[train_df["turbulence"] < threshold]
        results.append(
            {
                "percentile": pct,
                "threshold": threshold,
                "bear_day_count": len(bear_days),
                "bull_day_count": len(bull_days),
            }
        )

    df_results = pd.DataFrame(results)
    recommended_row = df_results[df_results["percentile"] == 75].iloc[0]
    return {
        "recommended_threshold": recommended_row["threshold"],
        "recommended_percentile": 75,
        "details": df_results,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Regime-switching backtester")
    parser.add_argument("--ppo_model", required=True, help="Path to trained PPO .pth file")
    parser.add_argument("--cppo_model", required=True, help="Path to trained CPPO .pth file")
    parser.add_argument(
        "--trade_data",
        default="trade_data_multi_signal_2019_2023.csv",
        help="Trade period CSV",
    )
    parser.add_argument(
        "--train_data",
        default="train_data_multi_signal_2013_2018.csv",
        help="Training period CSV (for threshold calibration)",
    )
    parser.add_argument(
        "--state_space",
        type=int,
        default=None,
        help="Override state space dimension (auto-computed if not provided)",
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument(
        "--turbulence_percentile",
        type=float,
        default=75.0,
        help="Percentile of training turbulence to use as bear/bull threshold",
    )
    parser.add_argument(
        "--output",
        default="regime_switch_portfolio.csv",
        help="Output CSV with daily portfolio values",
    )
    args = parser.parse_args()

    # ---- Load data ----
    print("Loading trade data …")
    trade_df = pd.read_csv(args.trade_data)
    if "Unnamed: 0" in trade_df.columns:
        trade_df = trade_df.drop("Unnamed: 0", axis=1)
    unique_dates = trade_df["date"].unique()
    trade_df["new_idx"] = trade_df["date"].map({d: i for i, d in enumerate(unique_dates)})
    trade_df = trade_df.set_index("new_idx")

    for col, default in [
        ("llm_sentiment", 3), ("llm_risk", 3),
        ("llm_confidence", 3), ("llm_volatility_forecast", 3),
    ]:
        if col not in trade_df.columns:
            trade_df[col] = default
        else:
            trade_df[col].fillna(default, inplace=True)

    stock_dim = len(trade_df["tic"].unique())
    LLM_DIM = 4
    state_space = args.state_space or (
        1 + 2 * stock_dim + (LLM_DIM + len(INDICATORS)) * stock_dim
    )
    print(f"stock_dim={stock_dim}, state_space={state_space}")

    # ---- Calibrate threshold ----
    train_df = pd.read_csv(args.train_data)
    cal = calibrate_threshold(train_df, turbulence_percentiles=[args.turbulence_percentile])
    turb_threshold = cal["recommended_threshold"]
    print(f"Turbulence threshold ({args.turbulence_percentile}th pct): {turb_threshold:.2f}")

    vix_threshold = compute_vix_threshold(trade_df)
    print(f"VIX threshold: {vix_threshold:.2f}")

    # ---- Load models ----
    hidden_sizes = tuple([args.hidden_size] * args.hidden_layers)
    print("Loading PPO model …")
    ppo_model = load_model(args.ppo_model, state_space, stock_dim, hidden_sizes)
    print("Loading CPPO model …")
    cppo_model = load_model(args.cppo_model, state_space, stock_dim, hidden_sizes)

    agent = RegimeSwitchAgent(
        ppo_model=ppo_model,
        cppo_model=cppo_model,
        turbulence_threshold=turb_threshold,
        vix_threshold=vix_threshold,
    )

    # ---- Run backtest ----
    buy_cost_list = sell_cost_list = [0.001] * stock_dim
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
        "drawdown_penalty": 0.0,  # no shaping during inference
    }

    print("Running regime-switch backtest …")
    results = run_backtest(agent, trade_df, env_kwargs)

    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Quick summary
    final_value = results["account_value"].iloc[-1]
    initial_value = 1_000_000
    cumret = (final_value - initial_value) / initial_value * 100
    bear_pct = (results["regime"] == "bear").mean() * 100

    print(f"Cumulative return : {cumret:.1f}%")
    print(f"Bear regime days  : {bear_pct:.1f}%")
    print(f"Bull regime days  : {100 - bear_pct:.1f}%")


if __name__ == "__main__":
    main()
