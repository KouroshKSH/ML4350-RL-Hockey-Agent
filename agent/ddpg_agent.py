import copy
import torch
import torch.nn.functional as F
from .networks import Actor, Critic
import numpy as np

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


class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        exploration_noise_type="gaussian",  # "gaussian" or "pink"
        pink_rows=16,
        seed=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

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

        if noise is not None and float(noise) != 0.0:
            if self.noise_type == "pink" and self._pink is not None:
                eps = self._pink.sample() * float(noise)
            else:
                eps = np.random.normal(0.0, float(noise), size=action.shape)
            action = (action + eps).clip(-1, 1)

        return action

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Critic Update
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.gamma * target_Q)

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update
        for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
        }