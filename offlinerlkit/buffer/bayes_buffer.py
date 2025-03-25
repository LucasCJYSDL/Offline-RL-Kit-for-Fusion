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
        self.full_next_observations = None
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        full_obss: np.ndarray,
        full_actions: np.ndarray,
        pre_actions: np.ndarray,
        time_steps: np.ndarray,
        hidden_states: np.ndarray,
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

        # initialize if necessary
        if self.full_observations is None:
            self.full_observations = np.zeros((self._max_size,) + full_obss.shape[1:], dtype=full_obss.dtype)
            self.full_actions = np.zeros((self._max_size, full_actions.shape[-1]), dtype=full_actions.dtype)
            self.pre_actions = np.zeros((self._max_size, pre_actions.shape[-1]), dtype=pre_actions.dtype)
            self.time_steps = np.zeros((self._max_size, 1), dtype=np.float32)
        
        if self.hidden_states is None:
            self.hidden_states = np.zeros((self._max_size,) + hidden_states.shape[1:], dtype=hidden_states.dtype)

        self.full_observations[indexes] = full_obss.copy()
        self.full_actions[indexes] = full_actions.copy()
        self.pre_actions[indexes] = pre_actions.copy()
        self.time_steps[indexes] = time_steps.copy()
        self.hidden_states[indexes] = hidden_states.copy()

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
            "full_observations": torch.tensor(self.full_observations[batch_indexes]).to(self.device),
            "full_actions": torch.tensor(self.full_actions[batch_indexes]).to(self.device),
            "pre_actions": torch.tensor(self.pre_actions[batch_indexes]).to(self.device),
            "time_steps": torch.tensor(self.time_steps[batch_indexes]).to(self.device),
            "hidden_states": torch.tensor(self.hidden_states[batch_indexes]).to(self.device),
            "priors": torch.tensor(self.priors[batch_indexes]).to(self.device)
        }        
    
    def sample_rollouts(self, batch_size, rollout_length) -> Dict[str, np.ndarray]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        f_act, time_step, terminal = [], [], []
        tmp_terminal = np.zeros_like(self.time_steps[batch_indexes])

        i = 0 
        batch_idx_list = []
        while not np.all(tmp_terminal):
            cur_batch_indexes = batch_indexes + i
            cur_batch_indexes[cur_batch_indexes >= self._size-1] = self._size-1
            f_act.append(self.full_actions[cur_batch_indexes].copy())
            time_step.append(self.time_steps[cur_batch_indexes].copy())
            batch_idx_list.append(cur_batch_indexes.copy())

            if i >= 1:
                tmp_terminal[time_step[-1] <= time_step[-2]] = 1
                terminal.append(tmp_terminal.copy())
            i += 1
            if i >= 100: # hacky, but we do not need very long lookahead
                break
        
        if i == 100:
            terminal[-1] = np.ones_like(self.time_steps[batch_indexes])
    
        ret = {
            "full_actions": np.moveaxis(np.array(f_act)[:-1], source=0, destination=1),
            "time_steps": np.moveaxis(np.array(time_step)[:-1], source=0, destination=1),
            "terminals": np.moveaxis(np.array(terminal, dtype=bool), source=0, destination=1),
            "priors": self.priors[batch_indexes].copy(),
            "full_observations": self.full_observations[batch_indexes].copy(),
            "pre_actions": self.pre_actions[batch_indexes].copy(),
            "hidden_states": np.moveaxis(self.hidden_states[batch_indexes], source=0, destination=2),
            "rollout_length": rollout_length,
            "batch_idx_list": batch_idx_list
        }
        
        return ret


    