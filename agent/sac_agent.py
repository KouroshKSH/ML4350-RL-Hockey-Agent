"""
Soft Actor-Critic (SAC) Agent Implementation
Developed by Kourosh Sharifi

Notes:
1. Implements an off-policy actor-critic algorithm that maximizes both expected reward and policy entropy for improved exploration.
2. Uses "Twin Critics" to reduce overestimation bias by using the minimum of two Q-networks during value updates.
3. Supports automated entropy tuning (auto-alpha) to maintain a target entropy level relative to the action dimension.
4. Stays consistent with the TQC agent interface for swapping during training and evaluation.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Critic
from .tqc_agent import GaussianPolicy


class SACAgent:
    """
    Soft Actor-Critic (continuous actions)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha="auto",
        init_alpha: float = 0.2,
        target_entropy=None,  # default: -action_dim
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = float(gamma)
        self.tau = float(tau)

        # actor (Gaussian, tanh-squashed)
        self.actor = GaussianPolicy(
            state_dim, 
            action_dim, 
            hidden_dim=int(actor_hidden_dim),
        ).to(self.device)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(actor_lr),
        )

        # twin critics + targets
        self.critic1 = Critic(
            state_dim, 
            action_dim, 
            hidden_dim=int(critic_hidden_dim),
        ).to(self.device)
        self.critic2 = Critic(
            state_dim, 
            action_dim, 
            hidden_dim=int(critic_hidden_dim),
        ).to(self.device)
        self.critic1_t = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_t = copy.deepcopy(self.critic2).to(self.device)

        self.critics_optim = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(critic_lr),
        )

        # entropy temperature to control exploration
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        # the lower alpha gets, the more deterministic the behavior becomes
        self.auto_alpha = (str(alpha).lower() == "auto")
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                float(np.log(init_alpha)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=float(actor_lr))
        else:
            self.log_alpha = None
            self.alpha_optim = None
            self.alpha = float(alpha)

    @torch.no_grad()
    def select_action(self, state, noise=0.1):
        """
        NOTE: in this repo, keep behavior consistent with TQC:
          - noise == 0  -> deterministic mean action
          - noise != 0  -> stochastic sampled action
        """
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        if noise is None or float(noise) == 0.0:
            # deterministic tanh(mean)
            _, _, a = self.actor.sample(s)
            return a.cpu().numpy().flatten()

        # stochastic sample
        a, _, _ = self.actor.sample(s)
        return a.cpu().numpy().flatten()

    def _soft_update(self, src: nn.Module, dst: nn.Module):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(int(batch_size))

        # Critic update
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_state)
            q1_next = self.critic1_t(next_state, next_action)
            q2_next = self.critic2_t(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)

            if self.auto_alpha:
                alpha_t = self.log_alpha.exp()
            else:
                alpha_t = torch.as_tensor(self.alpha, device=self.device, dtype=torch.float32)

            target_q = reward + not_done * self.gamma * (q_next - alpha_t * next_logp)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # actor update
        new_action, logp, egal = self.actor.sample(state)
        q1_pi = self.critic1(state, new_action)
        q2_pi = self.critic2(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        if self.auto_alpha:
            alpha_t = self.log_alpha.exp()
        else:
            alpha_t = torch.as_tensor(
                self.alpha, 
                device=self.device, 
                dtype=torch.float32,
            )

        actor_loss = (alpha_t * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha update
        alpha_loss_val = None
        if self.auto_alpha:
            # maximize entropy => minimize -alpha*(logp + target_entropy)
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_val = float(self.log_alpha.exp().detach().cpu().item())
            alpha_loss_val = float(alpha_loss.detach().cpu().item())
        else:
            alpha_val = float(self.alpha)

        # target update
        self._soft_update(self.critic1, self.critic1_t)
        self._soft_update(self.critic2, self.critic2_t)

        mean_q = float(q1.detach().mean().cpu().item())

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
            "mean_q": mean_q,
        }
