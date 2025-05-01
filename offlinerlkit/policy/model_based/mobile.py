import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from typing import Dict, Union, Tuple
from copy import deepcopy
from offlinerlkit.policy import BasePolicy


class MOBILEPolicy(BasePolicy):
    """
    Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning <Ref: https://proceedings.mlr.press/v202/sun23q.html>
    """

    def __init__(
        self,
        dynamics,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        state_idxs,
        action_idxs,
        sa_processor,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        num_samples: int = 10,
        deterministic_backup: bool = False,
        max_q_backup: bool = False
    ) -> None:

        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.critics = critics
        self.critics_old = deepcopy(critics)
        self.critics_old.eval()

        self.actor_optim = actor_optim
        self.critics_optim = critics_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._penalty_coef = penalty_coef
        self._num_samples = num_samples
        self._deteterministic_backup = deterministic_backup
        self._max_q_backup = max_q_backup

        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.sa_processor = sa_processor

    def train(self) -> None:
        self.actor.train()
        self.critics.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()
    
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

            # new for mobile
            rollout_transitions["hidden_states"].append(self.dynamics.get_memory()[idx_list])

            next_observations, rewards, terminals, info = self.dynamics.step(full_observations, pre_actions, full_actions, time_steps, time_terminals, self.state_idxs, init_samples["batch_idx_list"][t])
            next_observations = self.sa_processor.get_rl_state(next_observations, init_samples["batch_idx_list"][t+1])

            rollout_transitions["obss"].append(observations[idx_list])
            rollout_transitions["next_obss"].append(next_observations[idx_list])
            rollout_transitions["actions"].append(actions[idx_list])
            rollout_transitions["rewards"].append(rewards[idx_list])
            rollout_transitions["terminals"].append(terminals[idx_list])

            # new for mobile
            rollout_transitions["full_obss"].append(full_observations[idx_list])
            rollout_transitions["full_actions"].append(full_actions[idx_list])
            rollout_transitions["pre_actions"].append(pre_actions[idx_list])
            rollout_transitions["time_steps"].append(time_steps[idx_list])

            # new for rombrl
            rollout_transitions["time_terminals"].append(time_terminals[idx_list])
            rollout_transitions["batch_idx"].append(init_samples["batch_idx_list"][t][:, np.newaxis][idx_list]) # used to query the tracking targets
            rollout_transitions["next_batch_idx"].append(init_samples["batch_idx_list"][t+1][:, np.newaxis][idx_list])

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

    @ torch.no_grad()
    def compute_lcb(self, full_obss: torch.Tensor, pre_actions: torch.Tensor, full_actions: torch.Tensor, next_obss, hidden_states, terminals):
        # print(full_obss.shape, pre_actions.shape, full_actions.shape, time_steps.shape, hidden_states.shape)
        # compute next q std
        pred_next_obss = self.dynamics.sample_next_obss(full_obss, pre_actions, full_actions, hidden_states, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)

        pred_next_obss = pred_next_obss[:, self.state_idxs]
        # we implement the logic of sa_processor get_rl_state here - hacky
        subfix = next_obss[:, pred_next_obss.shape[1]:]
        targets = subfix[:, :subfix.shape[1]//2]
        targets = targets.unsqueeze(1).unsqueeze(1).repeat(num_samples, num_ensembles, 1, 1).reshape(-1, targets.shape[-1])
        differences = targets - pred_next_obss[:, self.sa_processor.idx_list]
        pred_next_obss = torch.cat([pred_next_obss, targets, differences], dim=-1)

        pred_next_actions, _ = self.actforward(pred_next_obss)
        
        pred_next_qs = torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old], 1)
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0) * (1 - terminals) # TODO: do not use termination signals here

        return penalty

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]

        # update critic
        qs = torch.stack([critic(obss, actions) for critic in self.critics], 0)
        with torch.no_grad():
            penalty = self.compute_lcb(mix_batch["full_observations"], mix_batch["pre_actions"], mix_batch["full_actions"], \
                                       mix_batch["next_observations"], mix_batch["hidden_states"], mix_batch["terminals"])
            penalty[:len(real_batch["rewards"])] = 0.0

            if self._max_q_backup:
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, 10, 1) \
                    .view(batch_size * 10, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_qs = torch.cat([critic_old(tmp_next_obss, tmp_next_actions) for critic_old in self.critics_old], 1)
                tmp_next_qs = tmp_next_qs.view(batch_size, 10, len(self.critics_old)).max(1)[0].view(-1, len(self.critics_old))
                next_q = torch.min(tmp_next_qs, 1)[0].reshape(-1, 1)
            else:
                next_actions, next_log_probs = self.actforward(next_obss)
                next_qs = torch.cat([critic_old(next_obss, next_actions) for critic_old in self.critics_old], 1)
                next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
                if not self._deteterministic_backup:
                    next_q -= self._alpha * next_log_probs
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((qs - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        qas = torch.cat([critic(obss, a) for critic in self.critics], 1)
        actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result