import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy


class MOPOPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        dynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        state_idxs,
        action_idxs,
        sa_processor,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.sa_processor = sa_processor
        self.dynamics = dynamics

    def rollout(
        self,
        init_samples
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        rollout_length = init_samples["full_observations"].shape[1]
        idx_list = np.array(range(0, init_samples["full_observations"].shape[0]))

        full_observations = init_samples["full_observations"][:, 0]
        full_actions = init_samples["full_actions"][:, 0]
        pre_actions = init_samples["pre_actions"][:, 0]
        time_steps = init_samples["time_steps"][:, 0]
        time_terminals = init_samples["terminals"][:, 0]

        self.dynamics.reset(init_samples["hidden_states"]) 

        for t in range(rollout_length):
            observations = full_observations[:, self.state_idxs]
            observations = self.sa_processor.get_rl_state(observations, init_samples["batch_idx_list"][t])

            actions = self.select_action(observations)
            step_actions = self.sa_processor.get_step_action(actions)
            full_actions[:, self.action_idxs] = step_actions.copy()

            next_observations, rewards, terminals, info = self.dynamics.step(full_observations, pre_actions, full_actions, time_steps, time_terminals, self.state_idxs, init_samples["batch_idx_list"][t])
            next_observations = self.sa_processor.get_rl_state(next_observations, init_samples["batch_idx_list"][t+1])

            rollout_transitions["obss"].append(observations[idx_list])
            rollout_transitions["next_obss"].append(next_observations[idx_list])
            rollout_transitions["actions"].append(actions[idx_list])
            rollout_transitions["rewards"].append(rewards[idx_list])
            rollout_transitions["terminals"].append(terminals[idx_list])

            num_transitions += len(idx_list)
            rewards_arr = np.append(rewards_arr, rewards[idx_list].flatten())

            nonterm_mask = (~terminals[idx_list]).flatten()
            if nonterm_mask.sum() == 0:
                break
            
            # danger
            full_observations = info["next_full_observations"]
            time_steps = info["next_time_steps"]
            pre_actions = full_actions.copy()
            if t < rollout_length - 1:
                idx_list = idx_list[nonterm_mask]
                full_actions = init_samples["full_actions"][:, t+1]
                time_terminals = init_samples["terminals"][:, t+1]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in fake_batch.keys()}
        return super().learn(mix_batch)
