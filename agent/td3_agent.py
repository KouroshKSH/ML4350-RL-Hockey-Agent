import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Actor


class PinkNoiseProcess:
    """
    Fast approximate 1/f ("pink") noise using the Voss-McCartney algorithm.
    """
    def __init__(self, size: int, num_rows: int = 16, seed=None):
        self.size = int(size)
        self.num_rows = int(num_rows)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.counter = 0
        self.rows = self.rng.standard_normal((self.num_rows, self.size)).astype(np.float32)

    def sample(self) -> np.ndarray:
        self.counter += 1
        c = self.counter
        tz = (c & -c).bit_length() - 1
        if tz >= self.num_rows:
            tz = self.num_rows - 1
        self.rows[: tz + 1] = self.rng.standard_normal((tz + 1, self.size)).astype(np.float32)
        return self.rows.sum(axis=0) / np.sqrt(self.num_rows)

class TwinCritic(nn.Module):
    """TD3 twin Q networks."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        return q1


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise_type="gaussian",  # "gaussian" or "pink"
        pink_rows=16,
        seed=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Twin critics
        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        # TD3 target policy smoothing
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.total_it = 0

        # Exploration noise for behavior policy
        self.noise_type = str(exploration_noise_type).lower()
        self.action_dim = int(action_dim)
        self._pink = None
        if self.noise_type == "pink":
            self._pink = PinkNoiseProcess(size=self.action_dim, num_rows=int(pink_rows), seed=seed)

    def reset_exploration(self):
        if self._pink is not None:
            self._pink.reset()

    def select_action(self, state, noise=0.1):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state_t).cpu().data.numpy().flatten()

        if noise is not None and float(noise) > 0.0:
            if self.noise_type == "pink" and self._pink is not None:
                eps = self._pink.sample() * float(noise)
            else:
                eps = np.random.normal(0.0, float(noise), size=action.shape)
            action = (action + eps).clip(-1, 1)

        return action

    def _soft_update(self, net, net_target):
        for p, p_targ in zip(net.parameters(), net_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss.detach().cpu().item())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": actor_loss_val,
        }