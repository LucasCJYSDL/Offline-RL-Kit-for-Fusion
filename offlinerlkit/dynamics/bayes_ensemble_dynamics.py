import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Tuple, Dict
from offlinerlkit.utils.scaler import StandardScaler
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
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim, scaler, terminal_fn, penalty_coef, uncertainty_mode)
        self._device = self.model.device
        self._sample_step = sample_step

    @ torch.no_grad()
    def step(
        self,
        prior: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray, 
        elite_only: bool,
        elite_list: np.ndarray,
        real_samples = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mus, logvars = self.model(obs_act)
        mus = mus.cpu()
        logvars = logvars.cpu()
        prior = torch.FloatTensor(prior)
        ensemble_size = prior.shape[0]
        prior_ls = prior.unsqueeze(-1).repeat(1, 1, mus.shape[-1])

        if elite_only:
            mus = mus[elite_list]
            logvars = logvars[elite_list]
        
        if self._sample_step:
            try:
                idx_list = torch.multinomial(prior.T, num_samples=1, replacement=True).T.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
            except:
                # print(prior.sum(dim=0))
                prior[:, prior.sum(dim=0) <= 0] = torch.tensor([1.0 / ensemble_size for _ in range(ensemble_size)], dtype=prior.dtype, device=prior.device).unsqueeze(1)
                idx_list = torch.multinomial(prior.T, num_samples=1, replacement=True).T.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
            mean_mus = torch.gather(mus, dim=0, index=idx_list).squeeze(0)
            ensemble_std = torch.gather(logvars.exp().sqrt(), dim=0, index=idx_list).squeeze(0)
        else:
            # GMM
            mean_mus = (prior_ls * mus).sum(dim=0)
            ensemble_var = (prior_ls * (logvars.exp() + mus * mus)).sum(dim=0) - mean_mus * mean_mus
            # ensemble_std = ensemble_var.sqrt() + 1e-6
            ensemble_std = ensemble_var.sqrt()
            ensemble_std[ensemble_std<=0] = 1e-6
            ensemble_std[torch.isnan(ensemble_std)] = 1e-6

        # sample
        dist = torch.distributions.Normal(mean_mus, ensemble_std)
        samples = dist.sample()
        samples_np = samples.numpy()
        next_obs = samples_np[..., :-1] + obs
        reward = samples_np[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        # print(next_obs.shape, reward.shape) # (50000, 17) (50000, 1)

        # get likelihood
        if real_samples is None:
            info["likelihood"] = get_prob(mus, logvars.exp().sqrt(), samples.unsqueeze(0).repeat(mus.shape[0], 1, 1)).clone().numpy() # torch.Size([7, 50000])
        else:
            real_samples = torch.FloatTensor(real_samples).to(mus.device)
            info["likelihood"] = get_prob(mus, logvars.exp().sqrt(), samples.unsqueeze(0).repeat(mus.shape[0], 1, 1)).clone().numpy()

        return next_obs, reward, terminal, info
    
    def _prepare_data(self, state, action, next_state, reward):
        # input
        delta_state = next_state - state
        state = torch.FloatTensor(state).to(self._device)
        action = torch.FloatTensor(action).to(self._device)
        state_act = torch.cat([state, action], dim=-1)
        state_act = self.scaler.transform_tensor(state_act)
        # output
        delta_state = torch.FloatTensor(delta_state).to(self._device)
        reward = torch.FloatTensor(reward).to(self._device).unsqueeze(-1)
        output = torch.cat([delta_state, reward], dim=-1)

        return state_act, output

    def get_bayes_priors(self, dataset):
        state, action, next_state, reward = dataset['observations'], dataset['actions'], \
                                            dataset['next_observations'], dataset['rewards']
        mini_batchsize = 5000
        total_len = state.shape[0]
        
        i = 0
        prob_ls = []
        while i * mini_batchsize < total_len:
            s_id = i * mini_batchsize
            e_id = min((i+1) * mini_batchsize, total_len)
            b_state, b_action, b_next_state, b_reward = state[s_id:e_id], action[s_id:e_id], \
                                                        next_state[s_id:e_id], reward[s_id:e_id]
            b_state_act, b_output = self._prepare_data(b_state, b_action, b_next_state, b_reward)

            mean, logvar = self.model(b_state_act)
            std = torch.sqrt(torch.exp(logvar))

            prob = get_prob(mean, std, b_output)
            # print(b_state_act.shape, b_output.shape, mean.shape, std.shape, prob.shape)
            # torch.Size([50000, 23]) torch.Size([50000, 18]) torch.Size([7, 50000, 18]) torch.Size([7, 50000, 18]) torch.Size([7, 50000])
            prob_ls.append(prob.cpu().detach().clone().numpy())
            i += 1
        
        return np.concatenate(prob_ls, axis=1)
    

    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int,
        is_bayes: bool = False
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))
        
        if not is_bayes:
            mean = mean[self.model.elites.data.cpu().numpy()] # torch.Size([7, 256, 18]) [5 3 6 4 1] torch.Size([5, 256, 18])
            std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss