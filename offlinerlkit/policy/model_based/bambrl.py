import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import MOBILEPolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.searcher import Searcher
from offlinerlkit.utils.scheduler import LinearParameter
from offlinerlkit.buffer import SLReplaBuffer, SL_Transition
from torch.distributions import Normal, Independent

class BAMBRLPolicy(MOBILEPolicy):

    def __init__(
        self,
        elite_only: bool,
        elite_list: bool,
        use_ba: bool,
        use_search: bool,
        search_ratio: float,
        sl_policy_only: bool,
        searcher: Searcher,
        sl_buffer: SLReplaBuffer,
        entropy_coe_scheduler: LinearParameter,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        num_samples: int = 10,
        deterministic_backup: bool = False,
        max_q_backup: bool = False
    ) -> None:

        super().__init__(dynamics, actor, critics, actor_optim, critics_optim, tau, gamma, 
                         alpha, penalty_coef, num_samples, deterministic_backup, max_q_backup)
        self._elite_only = elite_only
        self._elite_list = elite_list
        self._use_ba = use_ba
        self._use_search = use_search
        self._search_ratio = search_ratio
        self._sl_policy_only = sl_policy_only
        self._searcher = searcher
        self._sl_buffer = sl_buffer
        self._entropy_coe_scheduler = entropy_coe_scheduler
        self._device = self.actor.device
    
    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        return_dists: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        if not return_dists:
            return squashed_action, log_prob
        return squashed_action, log_prob, [dist.loc, dist.scale]

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        return_dists: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            if not return_dists:
                action, _ = self.actforward(obs, deterministic, return_dists)
            else:
                action, _, dists = self.actforward(obs, deterministic, return_dists)
        
        if not return_dists:
            return action.cpu().numpy()
        return action.cpu().numpy(), dists
    
    def _get_action(self, observations, priors):
        actions, logits = self.select_action(observations, return_dists=True)
        # using MCTS
        search_size = int(observations.shape[0] * self._search_ratio)
        # TODO: try different forms of randomness
        search_idx = np.random.choice(observations.shape[0], size=search_size, replace=False)
        # search_idx = np.arange(search_size)
        obs_input = observations[search_idx]
        prior_input = priors[:, search_idx]
        logits[0] = logits[0][search_idx].cpu().numpy()
        logits[1] = torch.nan_to_num(logits[1][search_idx], nan=1e-6).cpu().numpy()
    
        tree_roots = self._searcher.set_roots(search_size)
        self._searcher.prepare(tree_roots, prior_input, obs_input, logits)
        print("Start searching ...")
        self._searcher.search(tree_roots, self.get_search_quantity)

        print("Start sampling ...")
        searched_actions, action_dists, action_lists, q_list = self._searcher.sample(tree_roots)

        if self._sl_buffer is not None:
            self._sl_buffer.push(SL_Transition(obs_input, action_lists, [], action_dists, q_list))

        actions[search_idx] = searched_actions
        return actions
    
    @ torch.no_grad()
    def get_search_quantity(self, state_batch, action_batch, prior_batch, nextstate_batch):
        # get quantities for expansion 
        nextstate_batch = torch.FloatTensor(nextstate_batch).to(self._device)
        q_target, logprobs_batch, logits = self._get_next_q(nextstate_batch, return_dists=True)
        # -> numpy
        q_target = q_target.cpu().numpy()
        logprobs_batch = logprobs_batch.cpu().numpy()
        logits[0] = logits[0].cpu().numpy()
        logits[1] = logits[1].cpu().numpy()

        # get reward penalty
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        prior_batch = torch.FloatTensor(prior_batch).to(self._device)
        # penalty = self.compute_lcb(state_batch, action_batch, prior_batch)
        penalty = self.compute_lcb(state_batch, action_batch)
        
        if isinstance(self._alpha, float):
            augment = - self._alpha * self._gamma * logprobs_batch
        else:
            augment = - self._alpha.cpu().clone().numpy() * self._gamma * logprobs_batch

        return logits, q_target, augment, - self._penalty_coef * penalty.cpu().numpy()

    def rollout(
        self,
        init_obss: np.ndarray,
        init_priors: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        priors = init_priors.T
        for _ in range(rollout_length):
            if self._use_search:
                actions = self._get_action(observations, priors)
            else:
                actions = self.select_action(observations)
            # print(observations.shape, actions.shape, priors.shape)
            # (50000, 17) (50000, 6) (7, 50000)
            next_observations, rewards, terminals, info = self.dynamics.step(priors, observations, actions, self._elite_only, self._elite_list)
            # update the priors
            if self._use_ba:
                prods = info['likelihood'] * priors
                next_priors = prods / (prods.sum(axis=0, keepdims=True).repeat(prods.shape[0], axis=0) + 1e-6)
            else:
                next_priors = priors
            
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
            rollout_transitions["priors"].append(priors.T)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
            priors = next_priors[:, nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    # @ torch.no_grad()
    # def compute_lcb(self, obss: torch.Tensor, actions: torch.Tensor, priors: torch.Tensor):
    #     # compute next q std
    #     pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self._num_samples, is_bayes=True)
    #     num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
    #     pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
    #     pred_next_actions, _ = self.actforward(pred_next_obss)
        
    #     pred_next_qs = torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old], 1)
    #     pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
    #     values = pred_next_qs.mean(0).squeeze(-1).permute(1, 0) # (batch_size, n_ensemble)

    #     # get the std using the values and priors
    #     weighted_mean = torch.sum(values * priors, dim=1, keepdim=True).repeat(1, values.shape[1])
    #     variance = torch.sum(priors * (values - weighted_mean)**2, dim=1, keepdim=True)
    #     penalty = variance.sqrt()

    #     return penalty
    
    def _get_next_q(self, next_obss, return_dists=False):
        batch_size = next_obss.shape[0]
        if self._max_q_backup:
            tmp_next_obss = next_obss.unsqueeze(1) \
                .repeat(1, 10, 1) \
                .view(batch_size * 10, next_obss.shape[-1])
            
            if return_dists:
                tmp_next_actions, next_log_probs, next_dists = self.actforward(tmp_next_obss, return_dists=return_dists)
            else:
                tmp_next_actions, _ = self.actforward(tmp_next_obss, return_dists=return_dists)

            tmp_next_qs = torch.cat([critic_old(tmp_next_obss, tmp_next_actions) for critic_old in self.critics_old], 1)
            tmp_next_qs, tmp_next_q_idxes = tmp_next_qs.view(batch_size, 10, len(self.critics_old)).max(1)
            next_q, next_q_idx = torch.min(tmp_next_qs, 1)
            next_q, next_q_idx = next_q.reshape(-1, 1), next_q_idx.reshape(-1, 1)

            if return_dists:
                final_idx = torch.gather(tmp_next_q_idxes, dim=1, index=next_q_idx)
                final_idx = final_idx.unsqueeze(-1)
                next_log_probs = torch.gather(next_log_probs.view(batch_size, 10, -1), dim=1, index=final_idx).squeeze(1)
                final_idx = final_idx.repeat(1, 1, next_dists[0].shape[-1])
                next_dists[0] = torch.gather(next_dists[0].view(batch_size, 10, -1), dim=1, index=final_idx).squeeze(1)
                next_dists[1] = torch.gather(next_dists[1].view(batch_size, 10, -1), dim=1, index=final_idx).squeeze(1)
        else:
            if return_dists:
                next_actions, next_log_probs, next_dists = self.actforward(next_obss, return_dists=return_dists)
            else:
                next_actions, next_log_probs = self.actforward(next_obss, return_dists=return_dists)

            next_qs = torch.cat([critic_old(next_obss, next_actions) for critic_old in self.critics_old], 1)
            next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
            if (not self._deteterministic_backup) and (not return_dists):
                next_q -= self._alpha * next_log_probs
        
        if not return_dists:
            return next_q
        return next_q, next_log_probs, next_dists

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, rewards, terminals, priors = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], \
                                                               mix_batch["rewards"], mix_batch["terminals"], mix_batch["priors"]
        batch_size = obss.shape[0]
        # update critic
        qs = torch.stack([critic(obss, actions) for critic in self.critics], 0)
        with torch.no_grad():
            # penalty = self.compute_lcb(obss, actions, priors)
            penalty = self.compute_lcb(obss, actions)
            penalty[:len(real_batch["rewards"])] = 0.0 # ipt

            next_q = self._get_next_q(next_obss)
            
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((qs - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        if self._sl_buffer is None: 
            qas = torch.cat([critic(obss, a) for critic in self.critics], 1)
            actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        else:
            if self._sl_policy_only:
                actor_loss = self._sl_update(batch_size)
            else:
                actor_loss, sl_critic_loss = self._sl_update(batch_size)

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
        
        if not self._sl_policy_only:
            result["loss/sl_critic"] = sl_critic_loss.item()

        return result
    
    def _sl_update(self, batch_size):
        samples = self._sl_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(np.array(samples.state)).to(self._device)
        action_list_batch = torch.FloatTensor(np.array(samples.action_list)).to(self._device)
        action_dist_batch = torch.FloatTensor(np.array(samples.action_dist)).to(self._device)

        actor_loss_step, actor_ent_loss_step = self._sl_update_policy(state_batch, action_list_batch, action_dist_batch, samples.action_num)
        actor_loss = actor_loss_step + self._entropy_coe_scheduler.value * actor_ent_loss_step

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._entropy_coe_scheduler.decrease()

        if not self._sl_policy_only:
            q_batch = torch.FloatTensor(np.array(samples.q)).to(self._device)
            sl_critic_loss = self._sl_update_q_functions(state_batch, action_list_batch, action_dist_batch, q_batch)

            self.critics_optim.zero_grad()
            sl_critic_loss.backward()
            self.critics_optim.step()

            return actor_loss, sl_critic_loss

        return actor_loss

    def _sl_update_policy(self, state_batch, action_list_batch, action_dist_batch, action_num):
        max_action_num = int(np.max(action_num))

        _, logprobs_batch, logits = self.actforward(state_batch, return_dists=True)
        mu, std = logits[0], logits[1]
        dist = Independent(Normal(mu, std), 1)
        # print(mu.shape, std.shape, logprobs_batch.shape, mu.requires_grad, std.requires_grad, logprobs_batch.requires_grad)
        # torch.Size([256, 6]) torch.Size([256, 6]) torch.Size([256, 1]) True True True

        target_normalized_visit_count = action_dist_batch
        target_sampled_actions = action_list_batch
        # TODO: use logprobs_batch instead
        # policy_entropy_loss = -dist.entropy().mean()
        policy_entropy_loss = logprobs_batch.mean()

        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        log_prob_sampled_actions = []
        num_sampled_actions = target_normalized_visit_count.shape[-1]
        batch_size = target_normalized_visit_count.shape[0]

        for k in range(max_action_num):
            # SAC-like
            y = 1 - target_sampled_actions[:, k, :].pow(2)
            # NOTE: for numerical stability.
            min_val = torch.tensor(-1 + 1e-6).to(target_sampled_actions.device)
            max_val = torch.tensor(1 - 1e-6).to(target_sampled_actions.device)
            target_sampled_actions_clamped = torch.clamp(target_sampled_actions[:, k, :], min_val, max_val)
            target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)

            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(target_sampled_actions_before_tanh).unsqueeze(-1)
            # TODO: remove this lome
            log_prob = log_prob - torch.log(y + 1e-6).sum(-1, keepdim=True)
            log_prob = log_prob.squeeze(-1)
                
            log_prob_sampled_actions.append(log_prob)

        log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1) # torch.Size([256, 20])
    
        if max_action_num < num_sampled_actions:
            supplement = torch.zeros((batch_size, num_sampled_actions-max_action_num), dtype=torch.float32, device=target_sampled_actions.device)
            log_prob_sampled_actions = torch.cat([log_prob_sampled_actions, supplement], dim=-1)
        
        log_prob_sampled_actions[target_normalized_visit_count==0.0] = np.log(1e-6)
        
        # normalize the prob of sampled actions
        prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / (torch.exp(log_prob_sampled_actions).\
                                     sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach() + 1e-6)
        log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)
        # prepare the mask for optimization
        mask = (target_normalized_visit_count > 0.0).to(dtype=torch.float32, device=target_sampled_actions.device)

        # cross_entropy loss: - sum(p * log (q))
        policy_loss = - (torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions * mask).sum() / batch_size

        return policy_loss, policy_entropy_loss
    
    def _sl_update_q_functions(self, state_batch, action_list_batch, action_dist_batch, q_batch):
        # torch.Size([256, 11]) torch.Size([256, 10, 3]) torch.Size([256, 10]) torch.Size([256, 10])
        max_action_num = action_list_batch.shape[1]
        state_batch = state_batch.unsqueeze(1).repeat(1, max_action_num, 1).view(-1, state_batch.shape[-1])
        action_list_batch = action_list_batch.view(-1, action_list_batch.shape[-1])

        qs = torch.stack([critic(state_batch, action_list_batch) for critic in self.critics], 0) # torch.Size([2, 2560, 1])
        # note that q_batch has involved log_prob (as in update_q_functions) via the tree search backtraverse
        q_batch = q_batch.view(action_list_batch.shape[0], 1).unsqueeze(0).repeat(qs.shape[0], 1, 1)
        action_dist_batch = action_dist_batch.view(action_list_batch.shape[0], 1).unsqueeze(0).repeat(qs.shape[0], 1, 1)
        mask = (action_dist_batch > 0.0).to(dtype=torch.float32, device=self._device)
        
        q_loss = (torch.square(qs - q_batch) * mask).sum() / mask.sum()

        return q_loss