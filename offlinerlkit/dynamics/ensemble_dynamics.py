import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from typing import Tuple, Dict


class EnsembleDynamics:
    def __init__(
        self,
        model: nn.Module,
        terminal_fn,
        reward_fn,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        self.model = model
        self.terminal_fn = terminal_fn
        self.reward_fn = reward_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
    
    def reset(self, hidden_states=None):
        self.model.reset(hidden_states)

    def get_memory(self):
        return self.model.get_memory()

    @ torch.no_grad()
    def step(
        self,
        cur_state: np.ndarray,
        pre_action: np.ndarray,
        cur_action: np.ndarray,
        time_steps, 
        time_terminals, 
        state_idxs,
        batch_idxs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        info = {}
        net_input = np.concatenate([cur_state, pre_action, cur_action-pre_action], axis=-1).astype(np.float32)
        mean, std = self.model.forward(net_input)

        mean += cur_state
        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_member_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        info["next_full_observations"] = samples.copy()
        time_steps = time_steps + 1
        info["next_time_steps"] = time_steps.copy()

        # new for robust training
        info['dyn_net_input'] = net_input
        log_dyn_probs = norm.logpdf(ensemble_samples, mean, std)
        info['log_dyn_probs'] = log_dyn_probs[model_idxs, np.arange(batch_size)].sum(axis=-1)
        info['model_idxs'] = model_idxs
        info['dyn_samples'] = samples.copy() - cur_state

        # get the reward
        next_obs = samples[:, state_idxs]
        reward = self.reward_fn(next_obs, batch_idxs).reshape(-1, 1) # based on next state and current timesteps
        info["raw_reward"] = reward

        # get the termination signal
        terminal = self.terminal_fn(time_steps)
        terminal = terminal | time_terminals
        
        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            
            penalty = np.expand_dims(penalty, 1).astype(np.float32)

            assert penalty.shape == reward.shape, reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        full_obss, 
        pre_actions, 
        full_actions, 
        hidden_states,
        num_samples: int
    ) -> torch.Tensor:
        # reset the dynamics model
        if hidden_states is not None and int(full_obss.shape[0]) == int(hidden_states.shape[0]):
            self.reset(hidden_states=hidden_states.permute(1, 2, 0, 3))
        else:
            self.reset(hidden_states=hidden_states)

        net_input = torch.cat([full_obss, pre_actions, full_actions-pre_actions], dim=-1).to(torch.float32)
        mean, std = self.model.forward(net_input, is_tensor=True)

        next_obss = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0) # torch.Size([10, 5, 256, 27])

        return next_obss

