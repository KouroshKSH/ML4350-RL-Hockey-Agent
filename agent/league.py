import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

import hockey.hockey_env as h_env


def winner_to_payoff(winner: int) -> float:
    """Map env winner from agent1 perspective (1, 0, -1) to a zero-sum payoff."""
    if winner > 0:
        return 1.0
    if winner < 0:
        return -1.0
    return 0.0


class Policy:
    """Minimal policy interface for league evaluation."""

    def act_agent1(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def act_agent2(self, obs_agent2: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BotPolicy(Policy):
    def __init__(self, weak: bool):
        self.bot = h_env.BasicOpponent(weak=weak)

    def act_agent1(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros((4,), dtype=np.float32)

    def act_agent2(self, obs_agent2: np.ndarray) -> np.ndarray:
        return np.asarray(self.bot.act(obs_agent2), dtype=np.float32)


class ActorPolicy(Policy):
    """
    Wrap an actor network for use as either player1 or player2 in league games.

    - Uses obs_normalizer if provided (same normalizer used in training).
    - Uses deterministic action by default (tanh(mean) for GaussianPolicy).
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        device: torch.device,
        obs_normalizer=None,
        deterministic: bool = True,
    ):
        self.actor = actor
        self.device = device
        self.obs_normalizer = obs_normalizer
        self.deterministic = deterministic
        self.has_sample = hasattr(actor, "sample")

    def _act(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize_np(obs)
        s = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)

        if self.has_sample:
            if self.deterministic:
                _, _, a = self.actor.sample(s)  # tanh(mean)
            else:
                a, _, _ = self.actor.sample(s)
            return a.detach().cpu().numpy().flatten().astype(np.float32)

        a = self.actor(s)
        return a.detach().cpu().numpy().flatten().astype(np.float32)

    def act_agent1(self, obs: np.ndarray) -> np.ndarray:
        return self._act(obs)

    def act_agent2(self, obs_agent2: np.ndarray) -> np.ndarray:
        return self._act(obs_agent2)


def play_games(
    policy1: Policy,
    policy2: Policy,
    n_games: int,
    step_count: int,
    seed: Optional[int] = None,
) -> float:
    """
    Returns mean payoff from policy1 (as player1) perspective over n_games.
    payoff is in [-1, 1] (win=+1, tie=0, loss=-1).
    """
    env = h_env.HockeyEnv()
    if seed is not None:
        env.seed(seed)

    payoffs: List[float] = []
    for _g in range(int(n_games)):
        obs, _ = env.reset()
        obs2 = env.obs_agent_two()
        info = {}

        for _t in range(int(step_count)):
            a1 = policy1.act_agent1(obs)
            a2 = policy2.act_agent2(obs2)
            obs, _r, done, _tr, info = env.step(np.hstack([a1, a2]))
            obs2 = env.obs_agent_two()
            if done:
                break

        payoffs.append(winner_to_payoff(int(info.get("winner", 0))))

    return float(np.mean(payoffs)) if payoffs else 0.0


def symmetric_payoff(
    policy_i: Policy,
    policy_j: Policy,
    n_games: int,
    step_count: int,
    seed_base: int,
) -> float:
    """
    Role-averaged payoff for policy_i vs policy_j.

    (1) i as P1 vs j as P2 -> payoff_ij (from i perspective)
    (2) j as P1 vs i as P2 -> payoff_ji (from j perspective)

    When i plays as P2, i's payoff is (-payoff_ji). Average:
      payoff = 0.5 * (payoff_ij - payoff_ji)
    """
    payoff_ij = play_games(policy_i, policy_j, n_games=n_games, step_count=step_count, seed=seed_base)
    payoff_ji = play_games(policy_j, policy_i, n_games=n_games, step_count=step_count, seed=seed_base + 1)
    return 0.5 * (payoff_ij - payoff_ji)


def replicator_dynamics(A: np.ndarray, iters: int = 200, lr: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """
    Exponential-weights / replicator dynamics on an antisymmetric (or approximately antisymmetric) payoff matrix.
    Returns a probability vector over strategies (rows of A).
    """
    n = int(A.shape[0])
    p = np.ones(n, dtype=np.float64) / n
    for _ in range(int(iters)):
        pay = A @ p
        p = p * np.exp(lr * (pay - pay.max()))  # stability
        s = float(p.sum())
        if s <= eps:
            p = np.ones(n, dtype=np.float64) / n
        else:
            p /= s
    return p


@dataclass
class PSROConfig:
    enabled: bool = False
    start_episode: int = 3500
    update_interval_episodes: int = 600
    max_league_size: int = 6
    games_per_match: int = 3
    step_count: int = 250
    replicator_iters: int = 200
    replicator_lr: float = 2.0
    uniform_prob: float = 0.10


class PSROLeague:
    """
    PSRO-lite:
      - Every update interval, run a small round-robin among a limited subset of snapshots
      - Compute an approximate meta distribution via replicator dynamics
      - Provide meta_probs over snapshot_ids for sampling in training
    """
    def __init__(self, cfg: dict, device: torch.device):
        psro_cfg = (cfg.get("psro") or {})
        self.cfg = PSROConfig(
            enabled=bool(psro_cfg.get("enabled", False)),
            start_episode=int(psro_cfg.get("start_episode", 3500)),
            update_interval_episodes=int(psro_cfg.get("update_interval_episodes", 600)),
            max_league_size=int(psro_cfg.get("max_league_size", 6)),
            games_per_match=int(psro_cfg.get("games_per_match", 3)),
            step_count=int(psro_cfg.get("step_count", int(cfg.get("step_count", 250)))),
            replicator_iters=int(psro_cfg.get("replicator_iters", 200)),
            replicator_lr=float(psro_cfg.get("replicator_lr", 2.0)),
            uniform_prob=float(psro_cfg.get("uniform_prob", 0.10)),
        )
        self.device = device
        self.meta_probs: Dict[int, float] = {}
        self.last_update_episode: Optional[int] = None

    def _pick_league_entries(self, pool) -> List[dict]:
        snaps = list(getattr(pool, "snapshots", []))
        if not snaps:
            return []

        # prefer entries with "strength" defined, then most recent (higher id)
        def key(e):
            s = -1e9 if e.get("strength") is None else float(e["strength"])
            return (s, float(e.get("id", 0)))

        snaps.sort(key=key, reverse=True)
        k = min(len(snaps), int(self.cfg.max_league_size))
        return snaps[:k]

    @torch.no_grad()
    def maybe_update(self, episode: int, pool, obs_normalizer=None) -> Optional[Dict[int, float]]:
        if not self.cfg.enabled:
            return None
        if episode < self.cfg.start_episode:
            return None
        if self.cfg.update_interval_episodes <= 0:
            return None
        if episode % self.cfg.update_interval_episodes != 0:
            return None

        entries = self._pick_league_entries(pool)
        if len(entries) < 2:
            return None

        policies: List[ActorPolicy] = []
        ids: List[int] = []
        for e in entries:
            actor = e["actor_cls"](**e["actor_kwargs"]).to(self.device)
            actor.load_state_dict(e["state_dict"], strict=True)
            actor.eval()
            policies.append(
                ActorPolicy(actor, device=self.device, obs_normalizer=obs_normalizer, deterministic=True)
            )
            ids.append(int(e["id"]))

        n = len(policies)
        A = np.zeros((n, n), dtype=np.float64)

        # round-robin (upper triangle), role-averaged payoffs, enforce antisymmetry
        for i in range(n):
            for j in range(i + 1, n):
                seed_base = int(episode + 1000 * i + 10 * j)
                payoff = symmetric_payoff(
                    policy_i=policies[i],
                    policy_j=policies[j],
                    n_games=int(self.cfg.games_per_match),
                    step_count=int(self.cfg.step_count),
                    seed_base=seed_base,
                )
                A[i, j] = payoff
                A[j, i] = -payoff

        p = replicator_dynamics(A, iters=self.cfg.replicator_iters, lr=self.cfg.replicator_lr)
        p = np.asarray(p, dtype=np.float64)

        # mix with uniform for coverage
        if self.cfg.uniform_prob > 0:
            p = (1.0 - self.cfg.uniform_prob) * p + self.cfg.uniform_prob * (np.ones_like(p) / len(p))
            p = p / p.sum()

        self.meta_probs = {ids[i]: float(p[i]) for i in range(len(ids))}
        self.last_update_episode = int(episode)
        return dict(self.meta_probs)