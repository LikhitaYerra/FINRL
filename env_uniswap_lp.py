"""
Uniswap v3 Liquidity Provisioning RL Environment

Inspired by: "Uniswap liquidity provisioning with Reinforcement Learning"
             (Heimbach & Wattenhofer, 2022 + related work)

Task: An LP agent must decide:
  1. Which price range [pa, pb] to provide liquidity in
  2. How much capital to allocate vs hold
  3. When to rebalance (withdraw + re-deposit)

This extends the FinRL-DeepSeek framework to DeFi to demonstrate
that LLM signals (volatility forecasts, sentiment) generalise
beyond stock trading to decentralised exchange mechanics.

State space:
  - Current price P (normalised)
  - Price momentum (1d, 5d, 20d returns)
  - Pool volatility (rolling std of returns)
  - Bid-ask spread proxy (from Uniswap fee tier × volume ratio)
  - Agent's current position: [in_range?, center_price, width]
  - LLM signals: [sentiment, risk, confidence, volatility_forecast]
  - Portfolio composition: [LP_value, hold_value, fee_earnings]

Action space (continuous):
  - new_center ∈ [-1, 1] → price range center (normalised)
  - new_width  ∈ [0, 1]  → price range width (normalised)
  - allocate   ∈ [0, 1]  → fraction of portfolio to LP (rest held)

Reward = fee_income - impermanent_loss - rebalance_cost - risk_penalty

Usage:
    env = UniswapLPEnv(price_df)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ─── Uniswap v3 math ─────────────────────────────────────────────────────────

def in_range(P: float, pa: float, pb: float) -> bool:
    """True if price P is inside the LP range [pa, pb]."""
    return pa <= P <= pb


def liquidity_from_amounts(x: float, y: float, P: float,
                           pa: float, pb: float) -> float:
    """
    Uniswap v3: compute liquidity L given amounts x (token0) and y (token1).
    Simplified: assumes optimal split at current price.
    """
    if P <= pa:
        # All in token0
        sqP  = math.sqrt(P)
        sqPa = math.sqrt(pa)
        sqPb = math.sqrt(pb)
        return x * sqPa * sqPb / (sqPb - sqPa) if sqPb != sqPa else 0.0
    elif P >= pb:
        sqPb = math.sqrt(pb)
        sqPa = math.sqrt(pa)
        return y / (sqPb - sqPa) if sqPb != sqPa else 0.0
    else:
        sqP  = math.sqrt(P)
        sqPa = math.sqrt(pa)
        sqPb = math.sqrt(pb)
        # Optimal split formula
        Lx = x / (1/sqP - 1/sqPb) if sqPb != sqP else 0.0
        Ly = y / (sqP - sqPa) if sqP != sqPa else 0.0
        return min(Lx, Ly)


def amounts_from_liquidity(L: float, P: float,
                            pa: float, pb: float) -> tuple[float, float]:
    """
    Given L, compute amounts x and y needed at price P in range [pa, pb].
    """
    sqP  = math.sqrt(max(P, pa))
    sqPa = math.sqrt(pa)
    sqPb = math.sqrt(pb)
    sqPc = min(sqP, sqPb)

    x = L * (1/sqPc - 1/sqPb) if sqPb > sqPc else 0.0
    y = L * (sqPc - sqPa)     if sqPc > sqPa else 0.0
    return max(x, 0.0), max(y, 0.0)


def fee_earnings(L: float, P: float, pa: float, pb: float,
                 volume: float, fee_tier: float = 0.003) -> float:
    """
    Estimated fee earnings from a position.
    fee ∝ L × fee_tier × (fraction of day's volume attributed to this range)
    Simplified: LP earns fee_tier of the capital deployed per day.
    Realistic annual yield ≈ 5-30% for active LP in volatile pools.
    """
    if not in_range(P, pa, pb) or L <= 0:
        return 0.0
    # Daily yield: fee_tier × turnover_ratio (assume volume/L = 2 as turnover)
    daily_yield = fee_tier * 2.0
    return L * daily_yield / 365.0  # per-day earnings


def impermanent_loss(P0: float, P1: float, fraction_in_lp: float) -> float:
    """
    Standard IL formula: IL = 2√(P1/P0) / (1 + P1/P0) - 1
    Applied to the LP portion only.
    """
    if P0 <= 0 or P1 <= 0:
        return 0.0
    r = P1 / P0
    il = 2 * math.sqrt(r) / (1 + r) - 1  # always ≤ 0
    return il * fraction_in_lp


# ─── Environment ─────────────────────────────────────────────────────────────

LLM_COLS = ["llm_sentiment","llm_risk","llm_confidence","llm_volatility_forecast"]
N_LLM    = 4


class UniswapLPEnv(gym.Env):
    """
    Gymnasium environment for Uniswap v3 LP strategy.

    Observations (22-dim):
      [price_norm, ret_1d, ret_5d, ret_20d, vol_20d,
       in_range, center_norm, width_norm, allocate,
       lp_value_norm, hold_value_norm, fee_cumul_norm,
       drawdown_pct, total_return, step_frac,
       llm_sentiment, llm_risk, llm_confidence, llm_volatility,
       pool_volume_norm, fee_tier, spread_proxy]

    Actions (3-dim, clipped to [-1,1]):
      [new_center_norm, new_width_norm, new_allocate]
    """

    metadata = {"render_modes": []}
    OBS_DIM  = 22
    ACT_DIM  = 3

    def __init__(self,
                 price_df: pd.DataFrame,
                 initial_capital: float = 100_000.0,
                 fee_tier: float = 0.003,
                 rebalance_cost: float = 0.002,
                 reward_scaling: float = 1e-3,
                 drawdown_penalty: float = 0.1,
                 max_width_pct: float = 0.50,
                 max_width_min: float = 0.05,
                 ):
        """
        Args:
            price_df: DataFrame with columns [date, price, volume] and
                      optional LLM signal columns. One row per time step.
        """
        super().__init__()
        self.df            = price_df.reset_index(drop=True)
        self.initial_cap   = initial_capital
        self.fee_tier      = fee_tier
        self.rebalance_cost= rebalance_cost
        self.reward_scaling= reward_scaling
        self.drawdown_pen  = drawdown_penalty
        self.max_w_pct     = max_width_pct
        self.max_w_min     = max_width_min

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.ACT_DIM,), dtype=np.float32,
        )

        # Precompute price stats for normalisation
        prices        = self.df["price"].values.astype(float)
        self._P_mean  = prices.mean()
        self._P_std   = max(prices.std(), 1e-6)
        self._vol_base = np.std(np.diff(np.log(prices + 1e-9))) if len(prices) > 1 else 0.01

        # Volume normalisation
        vol_col = "volume" if "volume" in self.df.columns else None
        if vol_col:
            self._V_base = max(self.df[vol_col].mean(), 1e-6)
        else:
            self._V_base = 1.0

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t            = 0
        self.capital      = self.initial_cap
        row = self.df.iloc[0]
        self.P0           = float(row["price"])   # entry price
        self.center_norm  = 0.0      # centre of range (normalised)
        self.width_norm   = 0.2      # width (normalised)
        self.allocate     = 0.5      # start with 50% in LP to get reward signal

        self.lp_capital   = self.initial_cap * self.allocate
        self.hold_capital = self.initial_cap * (1 - self.allocate)
        self.fee_cumul    = 0.0
        self.peak_value   = self.initial_cap

        self._update_range()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        new_center_norm = float(action[0])
        new_width_norm  = float((action[1] + 1.0) / 2.0)  # [0,1]
        new_allocate    = float((action[2] + 1.0) / 2.0)  # [0,1]

        row  = self.df.iloc[self.t]
        P    = float(row["price"])
        ir_now = in_range(P, self.pa, self.pb)

        # Rebalance (if agent meaningfully changes position)
        cost = 0.0
        if (abs(new_center_norm - self.center_norm) > 0.05 or
                abs(new_width_norm - self.width_norm) > 0.05 or
                abs(new_allocate - self.allocate) > 0.05):
            total = self.lp_capital + self.hold_capital + self.fee_cumul
            cost  = total * self.rebalance_cost
            self.lp_capital   = (total - cost) * new_allocate
            self.hold_capital = (total - cost) * (1 - new_allocate)
            self.fee_cumul    = 0.0
            self.center_norm  = new_center_norm
            self.width_norm   = new_width_norm
            self.allocate     = new_allocate
            self._update_range()
            ir_now = in_range(P, self.pa, self.pb)

        # Fee income when in range (max possible: 30% annualised yield)
        max_daily_yield = 0.30 / 365.0   # 0.082% per day
        fees = self.lp_capital * max_daily_yield * (1.0 if ir_now else 0.0)
        self.fee_cumul += fees

        # Reward: fee income as fraction of capital (O(1e-3))
        # Penalty: rebalance cost + out-of-range opportunity cost
        out_of_range_penalty = self.lp_capital * max_daily_yield * 0.5 * (0 if ir_now else 1)
        reward = (fees - cost - out_of_range_penalty) / max(self.initial_cap, 1.0)

        total_val = self.lp_capital + self.hold_capital + self.fee_cumul
        self.peak_value = max(self.peak_value, total_val)
        drawdown_pct = (self.peak_value - total_val) / max(self.peak_value, 1.0)

        self.t += 1
        done = (self.t >= len(self.df) - 1)

        return self._obs(), float(reward), done, False, {
            "total_value": total_val,
            "fees": fees,
            "in_range": ir_now,
            "drawdown_pct": drawdown_pct,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_range(self):
        """Recompute pa, pb from current centre/width in normalised space."""
        P_cur    = float(self.df.iloc[self.t]["price"])
        center   = P_cur * (1 + self.center_norm * self.max_w_pct)
        hw       = max(center * self.width_norm * self.max_w_pct / 2,
                      P_cur * self.max_w_min)
        self.pa  = max(center - hw, 1e-9)
        self.pb  = center + hw

    def _obs(self) -> np.ndarray:
        t  = min(self.t, len(self.df) - 1)
        row = self.df.iloc[t]
        P   = float(row["price"])

        # Price features
        price_norm = (P - self._P_mean) / self._P_std
        ret_1d  = (P / float(self.df.iloc[max(t-1,0)]["price"]) - 1)
        ret_5d  = (P / float(self.df.iloc[max(t-5,0)]["price"]) - 1)
        ret_20d = (P / float(self.df.iloc[max(t-20,0)]["price"]) - 1)
        prices_win = self.df.iloc[max(t-20,0):t+1]["price"].values.astype(float)
        if len(prices_win) >= 2:
            log_rets = np.diff(np.log(prices_win + 1e-9))
            vol_20d  = float(np.std(log_rets))
        else:
            vol_20d  = self._vol_base

        # Position features
        ir_flag    = float(in_range(P, self.pa, self.pb))
        total_val  = self.lp_capital + self.hold_capital + self.fee_cumul

        # Normalised values
        lp_norm    = self.lp_capital / max(self.initial_cap, 1.0)
        hold_norm  = self.hold_capital / max(self.initial_cap, 1.0)
        fee_norm   = self.fee_cumul / max(self.initial_cap, 1.0)
        dd_pct     = (self.peak_value - total_val) / max(self.peak_value, 1.0)
        total_ret  = (total_val / self.initial_cap) - 1

        # Volume
        V_norm = (float(row["volume"]) / self._V_base
                  if "volume" in self.df.columns else 1.0)

        # LLM signals
        llm = [float(row.get(c, 3.0)) / 5.0  # normalise to [0,1]
               for c in LLM_COLS]

        obs = np.array([
            price_norm, ret_1d, ret_5d, ret_20d, vol_20d,
            ir_flag, self.center_norm, self.width_norm, self.allocate,
            lp_norm, hold_norm, fee_norm,
            dd_pct, total_ret, self.t / max(len(self.df), 1),
        ] + llm + [
            V_norm, self.fee_tier / 0.01, vol_20d / max(self._vol_base, 1e-9),
        ], dtype=np.float32)

        return obs


# ─── Synthetic data generator (for testing without real data) ─────────────────

def make_synthetic_pool(n_days: int = 500, seed: int = 42,
                        mu: float = 0.0003, sigma: float = 0.015,
                        start_price: float = 2000.0) -> pd.DataFrame:
    """
    Generate synthetic ETH/USDC-like price and volume data.
    Uses geometric Brownian motion with occasional jumps.
    """
    rng     = np.random.default_rng(seed)
    prices  = [start_price]
    for _ in range(n_days - 1):
        jump  = 1 + rng.choice([-0.05, 0, 0.05], p=[0.05, 0.90, 0.05])
        dP    = prices[-1] * (mu + sigma * rng.standard_normal()) * jump
        prices.append(max(prices[-1] + dP, 1.0))

    volumes = np.exp(rng.normal(8, 1, n_days)) * start_price / 1000
    dates   = pd.date_range("2020-01-01", periods=n_days, freq="D")

    df = pd.DataFrame({
        "date":   dates.strftime("%Y-%m-%d"),
        "price":  prices,
        "volume": volumes,
    })

    # Add neutral LLM signals
    for col in LLM_COLS:
        df[col] = 3.0

    return df


# ─── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing UniswapLPEnv …")
    df  = make_synthetic_pool(n_days=252)
    env = UniswapLPEnv(df, initial_capital=100_000)

    obs, _ = env.reset()
    print(f"  obs_dim={len(obs)}  expected={UniswapLPEnv.OBS_DIM}")

    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"  Episode: {steps} steps, total_reward={total_reward:.4f}")
    print(f"  Final: in_range={info['in_range']}, drawdown={info['drawdown_pct']:.2%}")
    print("UniswapLPEnv OK")
