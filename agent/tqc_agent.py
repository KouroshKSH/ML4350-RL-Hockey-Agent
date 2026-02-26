"""
Truncated Quantile Critics (TQC) Agent Implementation
Developed by Abtin Mogharabin

Notes:
1. Extends Soft Actor-Critic (SAC) by using Quantile Regression for the critics to model the full distribution of returns.
2. Implements "Truncated Quantile" estimation to control overestimation bias by dropping the top-most quantiles from multiple critics.
3. Features a Gaussian Policy with automated entropy tuning (alpha) and Huber loss for better quantile updates.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)

        # log prob with tanh correction
        log_prob = -0.5 * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        # deterministic action = tanh(mean)
        return action, log_prob, torch.tanh(mean)

class QuantileCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=25, hidden_dim=256):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)  # (B, n_quantiles)

def quantile_huber_loss(pred_quantiles, target_quantiles, taus, kappa=1.0):
    # target - pred
    diff = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # (B, N, K)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
    weight = (taus - (diff.detach() < 0).float()).abs()
    return (weight * huber).mean()

class TQCAgent:
    """
    TQC = SAC with quantile critics + truncation.
    keeps the same interface as your existing agents:
      - select_action(state, noise=...) -> np action
      - train(replay_buffer, batch_size) -> metrics dict
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        n_critics=5,
        n_quantiles=25,
        top_quantiles_to_drop=2, # per critic
        target_entropy=None, # default = -action_dim
        alpha="auto",
        init_alpha=0.2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critics = nn.ModuleList([
            QuantileCritic(state_dim, action_dim, n_quantiles=n_quantiles).to(self.device)
            for _ in range(n_critics)
        ])
        self.critics_target = copy.deepcopy(self.critics)
        self.critics_optim = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)

        self.gamma = float(gamma)
        self.tau = float(tau)

        self.n_critics = int(n_critics)
        self.n_quantiles = int(n_quantiles)
        self.top_quantiles_to_drop = int(top_quantiles_to_drop)
        self.total_quantiles = self.n_critics * self.n_quantiles
        self.drop_total = self.top_quantiles_to_drop * self.n_critics
        self.keep_total = self.total_quantiles - self.drop_total

        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        self.auto_alpha = (str(alpha).lower() == "auto")
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=actor_lr)
            self.alpha = float(init_alpha)
        else:
            self.alpha = float(alpha)
            self.log_alpha = None
            self.alpha_optim = None

        taus = (torch.arange(self.n_quantiles, device=self.device, dtype=torch.float32) + 0.5) / self.n_quantiles
        self.taus = taus.view(1, self.n_quantiles, 1)  # (1, N, 1)

    @torch.no_grad()
    def select_action(self, state, noise=0.1):
        """
        for TQC:
          - noise = 0 -> deterministic mean action
          - noise != 0 -> stochastic sample
        """
        s = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        if noise == 0 or noise is None:
            _, _, a = self.actor.sample(s)
            return a.cpu().numpy().flatten()
        a, _, _ = self.actor.sample(s)
        return a.cpu().numpy().flatten()

    def _soft_update(self, net, net_target):
        for p, p_t in zip(net.parameters(), net_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # critic update step
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_state)

            # target quantiles from all target critics
            tq_all = []
            for ct in self.critics_target:
                tq_all.append(ct(next_state, next_action))  # (B, N)
            tq_all = torch.cat(tq_all, dim=1)  # (B, n_critics*N)

            tq_all, _ = torch.sort(tq_all, dim=1)
            truncated = tq_all[:, :self.keep_total]  # keep lower quantiles

            alpha = self.alpha if not self.auto_alpha else float(self.log_alpha.exp().detach().cpu().item())
            alpha_t = torch.as_tensor(alpha, device=self.device, dtype=torch.float32)

            target = reward + not_done * self.gamma * (truncated - alpha_t * next_logp)

        critic_loss = 0.0
        q_means = []
        for c in self.critics:
            pred = c(state, action)  # (B, N)
            q_means.append(pred.mean().item())
            critic_loss = critic_loss + quantile_huber_loss(pred, target, self.taus)

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # actor's update step
        new_action, logp, _ = self.actor.sample(state)

        qs = []
        for c in self.critics:
            qs.append(c(state, new_action).mean(dim=1, keepdim=True))
        q_stack = torch.cat(qs, dim=1)  # (B, n_critics)
        q_min, _ = torch.min(q_stack, dim=1, keepdim=True)

        alpha = self.alpha if not self.auto_alpha else self.log_alpha.exp()
        actor_loss = (alpha * logp - q_min).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha update
        alpha_loss_val = None
        alpha_val = float(alpha.detach().cpu().item()) if torch.is_tensor(alpha) else float(alpha)

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_val = float(self.log_alpha.exp().detach().cpu().item())
            alpha_loss_val = float(alpha_loss.detach().cpu().item())

        # finally, update the target
        for c, ct in zip(self.critics, self.critics_target):
            self._soft_update(c, ct)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
            "mean_q": float(np.mean(q_means)) if q_means else 0.0,
        }
