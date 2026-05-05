"""
Multi-signal trading environment for FinRL Contest 2025 Task 1.

Extends env_stocktrading_llm_risk.py with two additional LLM signal dimensions:
  - llm_confidence           (1–5)
  - llm_volatility_forecast  (1–5)

State vector layout:
  [cash,
   price_1 … price_N,
   shares_1 … shares_N,
   indicator_1_1 … indicator_K_N,   (K indicators × N stocks)
   llm_sentiment_1 … llm_sentiment_N,
   llm_risk_1 … llm_risk_N,
   llm_confidence_1 … llm_confidence_N,
   llm_volatility_forecast_1 … llm_volatility_forecast_N]

state_space = 1 + 2*N + (len(INDICATORS) + 4)*N

Action modulation:
  - sentiment modulates buy/sell magnitude (inherited from base env)
  - high confidence amplifies the sentiment signal
  - high volatility_forecast tightens position sizes
"""

from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

matplotlib.use("Agg")


def _dummy_vec_env():
    """Lazy import — avoids pulling SB3 (and TensorBoard/TF paths) when unused."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv

_SIGNAL_COLS = [
    "llm_sentiment",
    "llm_risk",
    "llm_confidence",
    "llm_volatility_forecast",
]


class StockTradingEnv(gym.Env):
    """Multi-signal stock trading environment."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col: str = "turbulence",
        llm_sentiment_col: str = "llm_sentiment",
        llm_risk_col: str = "llm_risk",
        llm_confidence_col: str = "llm_confidence",
        llm_volatility_col: str = "llm_volatility_forecast",
        # Drawdown penalty coefficient – set in env_kwargs
        drawdown_penalty: float = 0.1,
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: list = [],
        model_name: str = "",
        mode: str = "",
        iteration: str = "",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.llm_sentiment_col = llm_sentiment_col
        self.llm_risk_col = llm_risk_col
        self.llm_confidence_col = llm_confidence_col
        self.llm_volatility_col = llm_volatility_col
        self.drawdown_penalty = drawdown_penalty
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.portfolio_peak = (
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        )
        self.asset_memory = [self.portfolio_peak]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.seed()

    # ---------------------------------------------------------------------- #
    # Trading mechanics (sell / buy)
    # ---------------------------------------------------------------------- #

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1]
                    * buy_num_shares
                    * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    # ---------------------------------------------------------------------- #
    # Step
    # ---------------------------------------------------------------------- #

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.asset_memory[0]
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_rewards.to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png"
                )
                plt.close()

            return self.state, self.reward, self.terminal, False, {}

        else:
            # ----------------------------------------------------------------
            # Multi-signal action modulation
            # ----------------------------------------------------------------
            llm_sentiments = self.data[self.llm_sentiment_col].values       # 1–5
            llm_confidences = self.data[self.llm_confidence_col].values     # 1–5
            llm_volatilities = self.data[self.llm_volatility_col].values    # 1–5

            buy_mask = actions > 0
            sell_mask = actions < 0

            strong_sell = llm_sentiments == 1
            moderate_sell = llm_sentiments == 2
            moderate_buy = llm_sentiments == 4
            strong_buy = llm_sentiments == 5

            # Base sentiment modulation (same as original)
            actions[(strong_sell & buy_mask) | (strong_buy & sell_mask)] *= 0.9
            actions[(moderate_sell & buy_mask) | (moderate_buy & sell_mask)] *= 0.95
            actions[(strong_sell & sell_mask) | (strong_buy & buy_mask)] *= 1.1
            actions[(moderate_sell & sell_mask) | (moderate_buy & buy_mask)] *= 1.05

            # Confidence amplification: high confidence → stronger signal
            confidence_scale = 1.0 + 0.02 * (llm_confidences - 3)  # ±0.04 at extremes
            actions *= confidence_scale

            # Volatility dampening: high volatility → reduce position sizes
            vol_scale = 1.0 - 0.05 * (llm_volatilities - 3)  # ±0.10 at extremes
            vol_scale = np.clip(vol_scale, 0.8, 1.2)
            actions *= vol_scale

            actions = actions * self.hmax
            actions = actions.astype(int)

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]

            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                else:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]

            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            # Update rolling peak for drawdown calculation
            self.portfolio_peak = max(self.portfolio_peak, end_total_asset)

            raw_reward = end_total_asset - begin_total_asset

            # Drawdown penalty: penalise being below rolling peak
            drawdown = (self.portfolio_peak - end_total_asset) / max(
                self.portfolio_peak, 1e-6
            )
            shaped_reward = raw_reward - self.drawdown_penalty * abs(raw_reward) * drawdown

            self.rewards_memory.append(raw_reward)
            self.reward = shaped_reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #

    def reset(self, *, seed=None, options=None):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            initial_portfolio = self.initial_amount + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
            self.asset_memory = [initial_portfolio]
            self.portfolio_peak = initial_portfolio
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
            self.portfolio_peak = previous_total_asset

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    # ---------------------------------------------------------------------- #
    # State construction
    # ---------------------------------------------------------------------- #

    def _initiate_state(self):
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                    + self.data[self.llm_sentiment_col].values.tolist()
                    + self.data[self.llm_risk_col].values.tolist()
                    + self.data[self.llm_confidence_col].values.tolist()
                    + self.data[self.llm_volatility_col].values.tolist()
                )
            else:
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.llm_sentiment_col]]
                    + [self.data[self.llm_risk_col]]
                    + [self.data[self.llm_confidence_col]]
                    + [self.data[self.llm_volatility_col]]
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                    + self.data[self.llm_sentiment_col].values.tolist()
                    + self.data[self.llm_risk_col].values.tolist()
                    + self.data[self.llm_confidence_col].values.tolist()
                    + self.data[self.llm_volatility_col].values.tolist()
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.llm_sentiment_col]]
                    + [self.data[self.llm_risk_col]]
                    + [self.data[self.llm_confidence_col]]
                    + [self.data[self.llm_volatility_col]]
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
                + self.data[self.llm_sentiment_col].values.tolist()
                + self.data[self.llm_risk_col].values.tolist()
                + self.data[self.llm_confidence_col].values.tolist()
                + self.data[self.llm_volatility_col].values.tolist()
            )
        else:
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                + [self.data[self.llm_sentiment_col]]
                + [self.data[self.llm_risk_col]]
                + [self.data[self.llm_confidence_col]]
                + [self.data[self.llm_volatility_col]]
            )
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            return self.data.date.unique()[0]
        return self.data.date

    # ---------------------------------------------------------------------- #
    # Memory helpers
    # ---------------------------------------------------------------------- #

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            df_date = pd.DataFrame(self.date_memory[:-1], columns=["date"])
            df_actions = pd.DataFrame(self.actions_memory)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            df_actions = pd.DataFrame(
                {"date": self.date_memory[:-1], "actions": self.actions_memory}
            )
        return df_actions

    def save_state_memory(self):
        date_list = self.date_memory[:-1]
        return pd.DataFrame({"date": date_list, "states": self.state_memory})

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        DummyVecEnv = _dummy_vec_env()
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
