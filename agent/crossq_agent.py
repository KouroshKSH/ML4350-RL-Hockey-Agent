import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class BatchRenorm1d(nn.Module):
    """
    Lightweight Batch Renormalization (Ioffe, 2017) for (B, F) MLP features.

    r,d corrections are clipped and warmed up over warmup_updates.
    This is used for CrossQ type stability when removing target networks.
    """
    def __init__(self, num_features: int, momentum: float = 0.01,
                 rmax: float = 3.0, dmax: float = 5.0, warmup_updates: int = 20000, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.rmax_final = float(rmax)
        self.dmax_final = float(dmax)
        self.warmup_updates = int(warmup_updates)
        self.eps = float(eps)

        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def _current_limits(self):
        if self.warmup_updates <= 0:
            return self.rmax_final, self.dmax_final
        t = float(self.num_updates.item())
        frac = min(max(t / float(self.warmup_updates), 0.0), 1.0)
        rmax = 1.0 + frac * (self.rmax_final - 1.0)
        dmax = 0.0 + frac * (self.dmax_final - 0.0)
        return rmax, dmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"BatchRenorm1d expects (B, F), got {tuple(x.shape)}")

        if self.training:
            self.num_updates += 1

            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_std = torch.sqrt(batch_var + self.eps)

            running_std = torch.sqrt(self.running_var + self.eps)

            rmax, dmax = self._current_limits()
            r = (batch_std / running_std).clamp(1.0 / rmax, rmax).detach()
            d = ((batch_mean - self.running_mean) / running_std).clamp(-dmax, dmax).detach()

            x_hat = ((x - batch_mean) / batch_std) * r + d

            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * batch_var)
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return x_hat * self.weight + self.bias


class CrossQGaussianPolicy(nn.Module):
    """
    GaussianPolicy with BatchRenorm + bounded activations (tanh).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 brn_momentum: float = 0.01, brn_rmax: float = 3.0, brn_dmax: float = 5.0, brn_warmup_updates: int = 20000):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.brn1 = BatchRenorm1d(hidden_dim, momentum=brn_momentum, rmax=brn_rmax, dmax=brn_dmax, warmup_updates=brn_warmup_updates)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.brn2 = BatchRenorm1d(hidden_dim, momentum=brn_momentum, rmax=brn_rmax, dmax=brn_dmax, warmup_updates=brn_warmup_updates)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.brn1(self.l1(state)))
        x = torch.tanh(self.brn2(self.l2(x)))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self(state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)

        log_prob = -0.5 * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class CrossQCritic(nn.Module):
    """
    Critic with BatchRenorm + bounded activations (tanh).
    Provides forward_joint(...) so normalization stats are computed on concatenated batches.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024,
                 brn_momentum: float = 0.01, brn_rmax: float = 3.0, brn_dmax: float = 5.0, brn_warmup_updates: int = 20000):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.brn1 = BatchRenorm1d(hidden_dim, momentum=brn_momentum, rmax=brn_rmax, dmax=brn_dmax, warmup_updates=brn_warmup_updates)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.brn2 = BatchRenorm1d(hidden_dim, momentum=brn_momentum, rmax=brn_rmax, dmax=brn_dmax, warmup_updates=brn_warmup_updates)
        self.l3 = nn.Linear(hidden_dim, 1)

    def _forward_x(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.brn1(self.l1(x)))
        x = torch.tanh(self.brn2(self.l2(x)))
        return self.l3(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self._forward_x(x)

    def forward_joint(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor):
        x1 = torch.cat([state, action], dim=1)
        x2 = torch.cat([next_state, next_action], dim=1)
        x = torch.cat([x1, x2], dim=0)
        q_all = self._forward_x(x)
        b = state.shape[0]
        return q_all[:b], q_all[b:]


class CrossQAgent:
    """
    CrossQ: A type of SAC agent with
      - NO target critics
      - BatchRenorm in actor+critic
      - critic joint forward pass
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        alpha="auto",
        init_alpha: float = 0.2,
        target_entropy=None,  # default: -action_dim
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 1024,
        brn_momentum: float = 0.01,
        brn_rmax: float = 3.0,
        brn_dmax: float = 5.0,
        brn_warmup_updates: int = 20000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(gamma)

        # Actor
        self.actor = CrossQGaussianPolicy(
            state_dim, action_dim, hidden_dim=int(actor_hidden_dim),
            brn_momentum=brn_momentum, brn_rmax=brn_rmax, brn_dmax=brn_dmax, brn_warmup_updates=int(brn_warmup_updates),
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))

        # Twin critics (NO targets)
        self.critic1 = CrossQCritic(
            state_dim, action_dim, hidden_dim=int(critic_hidden_dim),
            brn_momentum=brn_momentum, brn_rmax=brn_rmax, brn_dmax=brn_dmax, brn_warmup_updates=int(brn_warmup_updates),
        ).to(self.device)
        self.critic2 = CrossQCritic(
            state_dim, action_dim, hidden_dim=int(critic_hidden_dim),
            brn_momentum=brn_momentum, brn_rmax=brn_rmax, brn_dmax=brn_dmax, brn_warmup_updates=int(brn_warmup_updates),
        ).to(self.device)

        self.critics_optim = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(critic_lr),
        )

        # Temperature
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

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
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        if noise is None or float(noise) == 0.0:
            _, _, a = self.actor.sample(s)
            return a.cpu().numpy().flatten()
        a, _, _ = self.actor.sample(s)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(int(batch_size))

        # critic update (NO target networks)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_state)

            if self.auto_alpha:
                alpha_t = self.log_alpha.exp()
            else:
                alpha_t = torch.as_tensor(self.alpha, device=self.device, dtype=torch.float32)

        # Joint forward pass so BRN stats see both (s,a) and (s',a')
        q1, q1_next = self.critic1.forward_joint(state, action, next_state, next_action)
        q2, q2_next = self.critic2.forward_joint(state, action, next_state, next_action)

        with torch.no_grad():
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + not_done * self.gamma * (q_next - alpha_t * next_logp)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # actor update
        new_action, logp, _ = self.actor.sample(state)
        q1_pi = self.critic1(state, new_action)
        q2_pi = self.critic2(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        if self.auto_alpha:
            alpha_t2 = self.log_alpha.exp()
        else:
            alpha_t2 = torch.as_tensor(self.alpha, device=self.device, dtype=torch.float32)

        actor_loss = (alpha_t2 * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha update
        alpha_loss_val = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_val = float(self.log_alpha.exp().detach().cpu().item())
            alpha_loss_val = float(alpha_loss.detach().cpu().item())
        else:
            alpha_val = float(self.alpha)

        mean_q = float(q1.detach().mean().cpu().item())

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
            "mean_q": mean_q,
        }
