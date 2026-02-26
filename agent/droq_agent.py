import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tqc_agent import GaussianPolicy


class DroQCritic(nn.Module):
    """
    DroQ critic: MLP with LayerNorm + Dropout.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout_p: float = 0.01):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.do = nn.Dropout(p=float(dropout_p))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.do(F.relu(self.ln1(self.l1(x))))
        x = self.do(F.relu(self.ln2(self.l2(x))))
        return self.l3(x)


class DroQAgent:
    """
    DroQ: REDQ type agent with a small critic ensemble, where each critic is a dropout Q-function.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int = 2,
        target_subset: int = 2,
        dropout_p: float = 0.01,
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

        self.M = int(n_critics)
        self.N = int(target_subset)
        if self.N < 1 or self.N > self.M:
            raise ValueError(f"DroQ: target_subset must be in [1, n_critics], got {self.N} vs {self.M}")

        # Actor
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim=int(actor_hidden_dim)).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))

        # Dropout critic ensemble + targets
        self.critics = nn.ModuleList([
            DroQCritic(state_dim, action_dim, hidden_dim=int(critic_hidden_dim), dropout_p=float(dropout_p)).to(self.device)
            for _ in range(self.M)
        ])
        self.critics_t = copy.deepcopy(self.critics)
        self.critics_t.eval()
        self.critics_optim = torch.optim.Adam(self.critics.parameters(), lr=float(critic_lr))

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

        self._dropout_p = float(dropout_p)

    @torch.no_grad()
    def select_action(self, state, noise=0.1):
        """
        Consistent with SAC/TQC behavior in this repo:
          - noise == 0 => deterministic tanh(mean)
          - else       => stochastic sample
        """
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        if noise is None or float(noise) == 0.0:
            _, _, a = self.actor.sample(s)
            return a.cpu().numpy().flatten()
        a, _, _ = self.actor.sample(s)
        return a.cpu().numpy().flatten()

    def _soft_update(self, src: nn.Module, dst: nn.Module):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(int(batch_size))

        # critic update
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_state)

            if self.auto_alpha:
                alpha_t = self.log_alpha.exp()
            else:
                alpha_t = torch.as_tensor(self.alpha, device=self.device, dtype=torch.float32)

            idx = torch.randperm(self.M, device=self.device)[: self.N].tolist()
            q_subset = []
            for i in idx:
                q_subset.append(self.critics_t[i](next_state, next_action))
            q_subset = torch.cat(q_subset, dim=1)                          
            q_min = q_subset.min(dim=1, keepdim=True)[0]                   

            target_q = reward + not_done * self.gamma * (q_min - alpha_t * next_logp)

        critic_losses = []
        q_means = []
        for c in self.critics:
            q = c(state, action)
            q_means.append(float(q.detach().mean().cpu().item()))
            critic_losses.append(F.mse_loss(q, target_q))
        critic_loss = torch.stack(critic_losses).mean()

        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # actor update
        new_action, logp, _ = self.actor.sample(state)

        q_all = []
        for c in self.critics:
            q_all.append(c(state, new_action))
        q_all = torch.cat(q_all, dim=1)       
        q_pi = q_all.mean(dim=1, keepdim=True)

        if self.auto_alpha:
            alpha_t = self.log_alpha.exp()
        else:
            alpha_t = torch.as_tensor(self.alpha, device=self.device, dtype=torch.float32)

        actor_loss = (alpha_t * logp - q_pi).mean()

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

        # target update
        for c, ct in zip(self.critics, self.critics_t):
            self._soft_update(c, ct)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
            "mean_q": float(np.mean(q_means)) if q_means else 0.0,
            "dropout_p": self._dropout_p,
            "M": self.M,
            "N": self.N,
        }
