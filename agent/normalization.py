import numpy as np
import torch

class RunningMeanStd:
    """
    Welford running mean/std. Works with numpy arrays.
    Normalize: (x - mean) / std, with clipping.
    """
    def __init__(self, shape, epsilon=1e-4, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = float(tot_count)

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        if self.clip is not None:
            y = np.clip(y, -self.clip, self.clip)
        return y

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        y = (x - mean) / std
        if self.clip is not None:
            y = torch.clamp(y, -self.clip, self.clip)
        return y