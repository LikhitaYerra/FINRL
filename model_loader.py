"""
Shared model definition that matches the saved checkpoint format.

All evaluation scripts import from here to avoid repeating the architecture
and ensuring compatibility with the saved state dicts.
"""

import numpy as np
import torch
import torch.nn as nn

from signal_attention import SAMStateEncoder


def _mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.log_std = nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)))
        self.mu_net  = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return self.mu_net(obs), torch.exp(self.log_std)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512, 512), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            mu, _ = self.pi(obs)
        return mu.numpy()


def load_cppo_model(path: str, obs_dim: int, act_dim: int) -> MLPActorCritic:
    """Load a saved CPPO model from disk."""
    ac = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    ac.load_state_dict(torch.load(path, map_location="cpu"))
    ac.eval()
    return ac


class SAMMPIActorCritic(nn.Module):
    """
    Mirrors SAMActorCriticTraining from train_cppo_sam.py for inference only.
    Checkpoint keys: encoder.*, pi_net.*, v_net.*, log_std
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(512, 512),
        activation=nn.ReLU,
        n_stocks: int = 30,
        n_tech: int = 8,
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
        self.log_std = nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)))
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.pi_net = nn.Sequential(*layers)

        v_sizes = [obs_dim] + list(hidden_sizes) + [1]
        v_layers = []
        for j in range(len(v_sizes) - 1):
            act = activation if j < len(v_sizes) - 2 else nn.Identity
            v_layers += [nn.Linear(v_sizes[j], v_sizes[j + 1]), act()]
        self.v_net = nn.Sequential(*v_layers)

    def act(self, obs):
        with torch.no_grad():
            enc = self.encoder(obs)
            mu = self.pi_net(enc)
        return mu.numpy()


def load_sam_mpi_cppo(
    path: str,
    obs_dim: int,
    act_dim: int,
    n_stocks: int = 30,
    n_tech: int = 8,
    n_signals: int = 4,
) -> SAMMPIActorCritic:
    ac = SAMMPIActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_stocks=n_stocks,
        n_tech=n_tech,
        n_signals=n_signals,
    )
    ac.load_state_dict(torch.load(path, map_location="cpu"))
    ac.eval()
    return ac
