"""
Signal Attention Module (SAM)

Learns to contextually weight each of the four LLM signal dimensions
based on the current market state (technical indicators + turbulence).

Architecture:
  - Query: market context vector  (projected from tech features)
  - Key/Value: LLM signal matrix  (4 signals × N stocks)
  - Output: attended signal vector, concat'd with original state

This replaces simple concatenation with a cross-attention mechanism
that lets the policy learn WHEN each signal type matters:
  - Sentiment matters more in trending markets
  - Risk signal matters more near drawdown peaks
  - Volatility signal matters more when turbulence is rising
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core cross-attention block
# ─────────────────────────────────────────────────────────────────────────────

class SignalAttentionModule(nn.Module):
    """
    Cross-attention between market context and LLM signal dimensions.

    Args:
        n_stocks:     Number of stocks (30 for NASDAQ-100)
        n_signals:    Number of LLM signal dimensions (4)
        n_tech:       Number of technical indicators per stock (7)
        d_model:      Internal attention dimension
        n_heads:      Number of attention heads
    """

    def __init__(
        self,
        n_stocks: int = 30,
        n_signals: int = 4,
        n_tech: int = 10,  # n_indicators + 2 (price + holdings), passed by SAMStateEncoder
        d_model: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_stocks  = n_stocks
        self.n_signals = n_signals
        self.d_model   = d_model

        # Project per-stock context (tech + price + holdings) → query
        self.query_proj = nn.Linear(n_tech, d_model)

        # Project each LLM signal dimension → key/value
        self.key_proj   = nn.Linear(n_stocks, d_model)
        self.value_proj = nn.Linear(n_stocks, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # Project output back to n_signals dimensions (per stock)
        self.out_proj = nn.Linear(d_model, n_stocks)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(n_stocks)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tech_features: torch.Tensor,   # (B, n_stocks, n_tech+2)
        llm_signals: torch.Tensor,      # (B, n_signals, n_stocks)
    ) -> torch.Tensor:
        """
        Returns attended signals: (B, n_signals, n_stocks)

        The attention weights tell us which signal dimensions the
        market context is currently ``asking about''.
        """
        B = tech_features.shape[0]

        # Query: aggregate across stocks → (B, 1, d_model)
        q = self.query_proj(tech_features)          # (B, n_stocks, d_model)
        q = q.mean(dim=1, keepdim=True)             # (B, 1, d_model)

        # Keys/Values: one vector per signal dimension → (B, n_signals, d_model)
        k = self.key_proj(llm_signals)              # (B, n_signals, d_model)
        v = self.value_proj(llm_signals)            # (B, n_signals, d_model)

        # Cross-attention: (B, 1, d_model)
        attn_out, attn_weights = self.attn(q, k, v)  # attn_weights (B, 1, n_signals)
        attn_out = attn_out.expand(-1, self.n_signals, -1)  # (B, n_signals, d_model)

        # Project back to n_stocks
        out = self.out_proj(attn_out)               # (B, n_signals, n_stocks)
        out = self.norm(out + llm_signals)          # residual connection

        return out, attn_weights.squeeze(1)         # (B, n_signals, n_stocks), (B, n_signals)


# ─────────────────────────────────────────────────────────────────────────────
# State encoder that integrates SAM into the full observation
# ─────────────────────────────────────────────────────────────────────────────

class SAMStateEncoder(nn.Module):
    """
    Replaces raw state concatenation with attention-weighted signals.

    Input  : raw observation vector of dim STATE_SPACE
    Output : enhanced observation of dim STATE_SPACE
              (same shape, drop-in for existing MLP policy)

    State layout assumed (from env_stocktrading_multi_signal):
      [cash(1), prices(N), holdings(N), tech(K*N), signals(4*N)]
    """

    def __init__(
        self,
        n_stocks: int = 30,
        n_tech: int = 7,
        n_signals: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_stocks  = n_stocks
        self.n_tech    = n_tech
        self.n_signals = n_signals

        # Indices in the flat state vector
        self.cash_end     = 1
        self.price_end    = 1 + n_stocks
        self.hold_end     = 1 + 2 * n_stocks
        self.tech_end     = 1 + 2 * n_stocks + n_tech * n_stocks
        self.signal_end   = self.tech_end + n_signals * n_stocks

        self.sam = SignalAttentionModule(
            n_stocks=n_stocks,
            n_signals=n_signals,
            n_tech=n_tech + 2,  # indicators + price + holdings per stock
            d_model=d_model,
            n_heads=n_heads,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, STATE_SPACE) or (STATE_SPACE,) flat vector
        Returns: same shape as obs, with signals replaced by attended signals
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        B = obs.shape[0]
        N, K, S = self.n_stocks, self.n_tech, self.n_signals

        # Parse state components
        prices   = obs[:, self.cash_end : self.price_end]         # (B, N)
        holdings = obs[:, self.price_end : self.hold_end]         # (B, N)
        tech_flat= obs[:, self.hold_end : self.tech_end]          # (B, K*N)
        sig_flat = obs[:, self.tech_end : self.signal_end]        # (B, 4*N)

        # Reshape for attention
        tech     = tech_flat.view(B, N, K)                        # (B, N, K)
        # Add price and holdings as extra "tech" features per stock
        ph       = torch.stack([prices, holdings], dim=-1)        # (B, N, 2)
        tech_ext = torch.cat([tech, ph], dim=-1)                  # (B, N, K+2)

        signals  = sig_flat.view(B, S, N)                         # (B, S, N)

        # Attend
        attended, _ = self.sam(tech_ext, signals)                 # (B, S, N)
        attended_flat = attended.view(B, S * N)                   # (B, S*N)

        # Reconstruct state with attended signals
        enhanced = torch.cat([
            obs[:, :self.tech_end],   # cash + prices + holdings + tech (unchanged)
            attended_flat,            # attended signals
        ], dim=1)

        if squeeze:
            enhanced = enhanced.squeeze(0)

        return enhanced


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in Actor-Critic with SAM
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SAMActorCritic(nn.Module):
    """
    MLPActorCritic enhanced with a Signal Attention Module.

    The SAM pre-processes the observation before it reaches the policy/value
    networks, replacing raw signal concatenation with attended signals.
    This is a drop-in replacement for MLPActorCritic in train_cppo_multi_signal.py.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(512, 512),
        activation=nn.Tanh,
        n_stocks: int = 30,
        n_tech: int = 8,   # len(INDICATORS) = 8 for NASDAQ-100 setup
        n_signals: int = 4,
        sam_d_model: int = 64,
        sam_n_heads: int = 4,
    ):
        super().__init__()

        self.encoder = SAMStateEncoder(
            n_stocks=n_stocks,
            n_tech=n_tech,
            n_signals=n_signals,
            d_model=sam_d_model,
            n_heads=sam_n_heads,
        )

        # Policy network
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.pi = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

        # Value network
        self.v = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def act(self, obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            encoded = self.encoder(obs)
            mu = self.pi(encoded)
        return mu.numpy()

    def step(self, obs: torch.Tensor):
        with torch.no_grad():
            encoded = self.encoder(obs)
            mu  = self.pi(encoded)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            a   = dist.sample()
            logp = dist.log_prob(a).sum(axis=-1)
            v   = self.v(encoded)
        return a.numpy(), v.numpy(), logp.numpy()

    def forward_with_attention(self, obs: torch.Tensor):
        """Diagnostic: returns attention weights for interpretability."""
        encoded = self.encoder(obs)
        tech_start = self.encoder.hold_end
        tech_end   = self.encoder.tech_end
        sig_end    = self.encoder.signal_end
        N, K, S    = self.encoder.n_stocks, self.encoder.n_tech, self.encoder.n_signals
        B          = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        prices   = obs[:, 1:1+N]
        holdings = obs[:, 1+N:1+2*N]
        tech     = obs[:, tech_start:tech_end].view(B, N, K)
        ph       = torch.stack([prices, holdings], dim=-1)
        tech_ext = torch.cat([tech, ph], dim=-1)
        signals  = obs[:, tech_end:sig_end].view(B, S, N)
        _, weights = self.encoder.sam(tech_ext, signals)
        return weights  # (B, n_signals)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from finrl.config import INDICATORS
    N, K, S = 30, len(INDICATORS), 4
    obs_dim = 1 + 2*N + K*N + S*N  # 421

    model = SAMActorCritic(obs_dim=obs_dim, act_dim=N)
    print(f"SAMActorCritic parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_obs = torch.randn(8, obs_dim)  # batch of 8
    encoded   = model.encoder(dummy_obs)
    print(f"Encoder input:  {dummy_obs.shape}")
    print(f"Encoder output: {encoded.shape}")

    weights = model.forward_with_attention(dummy_obs)
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights (first sample): {weights[0].detach().numpy().round(3)}")

    a, v, logp = model.step(dummy_obs[0])
    print(f"Action shape: {a.shape}, Value: {float(v):.4f}")
    print("SAM smoke test PASSED")
