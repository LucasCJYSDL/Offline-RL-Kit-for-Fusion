from offlinerlkit.buffer import ReplayBuffer
from typing import Tuple, Dict
import numpy as np
import torch

class BayesReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        prior_dim: int,
        device: str = "cpu"
    ) -> None:
        super().__init__(buffer_size, obs_shape, obs_dtype, action_dim, action_dtype, device)
        self.priors = np.zeros((self._max_size, prior_dim), dtype=np.float32)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        priors: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()
        self.priors[indexes] = np.array(priors).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_prior(self, priors: np.ndarray) -> None:
        assert len(priors) == self._size
        self.priors = priors
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device), 
            "priors": torch.tensor(self.priors[batch_indexes]).to(self.device)
        }


    