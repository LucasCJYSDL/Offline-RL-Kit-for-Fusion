import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Tuple, Dict
from offlinerlkit.dynamics.ensemble_dynamics import EnsembleDynamics


def get_prob(mean, std, b_output):
    dist = torch.distributions.Normal(mean, std)
    l_prob = dist.log_prob(b_output) # log of pdf
    prob = l_prob.mean(-1).exp() # not sum, to normalize with the action dim

    return prob

class BayesEnsembleDynamics(EnsembleDynamics):
    def __init__(
        self,
        sample_step: bool,
        model: nn.Module,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        reward_fn,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, terminal_fn, reward_fn, penalty_coef, uncertainty_mode)
        self._device = self.model.device
        self._sample_step = sample_step

    @ torch.no_grad()
    def step(
        self,
        prior: np.ndarray,
        cur_state: np.ndarray,
        pre_action: np.ndarray,
        cur_action: np.ndarray,
        time_steps, 
        time_terminals, 
        state_idxs,
        batch_idxs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        info = {}
        net_input = np.concatenate([cur_state, pre_action, cur_action-pre_action], axis=-1).astype(np.float32)
        mus, stds = self.model.forward(net_input)
        mus, stds = torch.FloatTensor(mus), torch.FloatTensor(stds)

        prior = torch.FloatTensor(prior)
        ensemble_size = prior.shape[0]
        prior_ls = prior.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
        
        if self._sample_step:
            try:
                idx_list = torch.multinomial(prior.T, num_samples=1, replacement=True).T.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
            except:
                # print(prior.sum(dim=0))
                prior[:, prior.sum(dim=0) <= 0] = torch.tensor([1.0 / ensemble_size for _ in range(ensemble_size)], dtype=prior.dtype, device=prior.device).unsqueeze(1)
                idx_list = torch.multinomial(prior.T, num_samples=1, replacement=True).T.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
            mean_mus = torch.gather(mus, dim=0, index=idx_list).squeeze(0)
            ensemble_std = torch.gather(stds, dim=0, index=idx_list).squeeze(0)
        else:
            # GMM
            mean_mus = (prior_ls * mus).sum(dim=0)
            ensemble_var = (prior_ls * (stds.square() + mus * mus)).sum(dim=0) - mean_mus * mean_mus
            ensemble_std = ensemble_var.sqrt()
            ensemble_std[ensemble_std<=0] = 1e-6
            ensemble_std[torch.isnan(ensemble_std)] = 1e-6

        # sample
        dist = torch.distributions.Normal(mean_mus, ensemble_std)
        samples = dist.sample()
        samples_np = samples.numpy() + cur_state

        info["next_full_observations"] = samples_np.copy()
        time_steps = time_steps + 1
        info["next_time_steps"] = time_steps.copy()

        # get the reward
        next_obs = samples_np[:, state_idxs]
        reward = self.reward_fn(next_obs, batch_idxs).reshape(-1, 1)
        info["raw_reward"] = reward

        # get the termination signal
        terminal = self.terminal_fn(time_steps)
        terminal = terminal | time_terminals

        # get likelihood
        info["likelihood"] = get_prob(mus, stds, samples.unsqueeze(0).repeat(mus.shape[0], 1, 1)).clone().numpy() # torch.Size([7, 50000])

        return next_obs, reward, terminal, info
    
    def _prepare_data(self, state, pre_action, action, next_state):
        # input
        delta_action = action - pre_action
        state_act = torch.cat([torch.FloatTensor(state), torch.FloatTensor(pre_action), torch.FloatTensor(delta_action)], dim=-1).to(self._device)
        # output
        delta_state = next_state - state
        output = torch.FloatTensor(delta_state).to(self._device)

        return state_act, output

    def get_bayes_priors(self, dataset):
        state, pre_action, action, next_state, hidden_states = dataset['full_observations'], dataset['pre_actions'],\
                                                               dataset['full_actions'], dataset['full_next_observations'],\
                                                               np.moveaxis(dataset['hidden_states'], source=0, destination=2)
        mini_batchsize = 5000
        total_len = state.shape[0]
        
        i = 0
        prob_ls = []
        while i * mini_batchsize < total_len:
            s_id = i * mini_batchsize
            e_id = min((i+1) * mini_batchsize, total_len)
            b_state, b_pre_action, b_action, b_next_state = state[s_id:e_id], pre_action[s_id:e_id], \
                                                            action[s_id:e_id], next_state[s_id:e_id]
            b_state_act, b_output = self._prepare_data(b_state, b_pre_action, b_action, b_next_state)

            b_hidden_states = hidden_states[:, :, s_id:e_id]
            self.model.reset(hidden_states=b_hidden_states)
            mean, std = self.model(b_state_act, is_tensor=True)

            prob = get_prob(mean, std, b_output)
            # torch.Size([50000, 23]) torch.Size([50000, 18]) torch.Size([7, 50000, 18]) torch.Size([7, 50000, 18]) torch.Size([7, 50000])
            prob_ls.append(prob.cpu().detach().clone().numpy())
            i += 1

        return np.concatenate(prob_ls, axis=1)
    