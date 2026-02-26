import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_normalizer = None

    def set_obs_normalizer(self, normalizer):
        self.obs_normalizer = normalizer

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        s = torch.as_tensor(self.state[ind], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(self.action[ind], dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(self.next_state[ind], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(self.reward[ind], dtype=torch.float32, device=self.device)
        nd = torch.as_tensor(self.not_done[ind], dtype=torch.float32, device=self.device)

        if self.obs_normalizer is not None:
            s = self.obs_normalizer.normalize_torch(s)
            ns = self.obs_normalizer.normalize_torch(ns)

        return s, a, ns, r, nd
