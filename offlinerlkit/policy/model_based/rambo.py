import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from operator import itemgetter
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.buffer import ModelSLReplayBuffer


class RAMBOPolicy(MOPOPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
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
        dynamics_adv_optim: torch.optim.Optimizer,
        state_idxs,
        action_idxs,
        sa_processor,
        model_sl_buffer: ModelSLReplayBuffer,
        update_hidden_states,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        adv_weight: float = 0,
        adv_train_steps: int = 500,
        adv_rollout_batch_size: int = 256,
        adv_rollout_length: int = 5,
        sl_batch_size: int = 16,
        include_ent_in_adv: bool = False,
        scaler: StandardScaler = None,
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics, 
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            state_idxs,
            action_idxs,
            sa_processor,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self.model_sl_buffer = model_sl_buffer
        self._dynmics_adv_optim = dynamics_adv_optim
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._sl_batch_size = sl_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self._update_hidden_states = update_hidden_states
        self.scaler = scaler
        self.device = device
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "rambo_pretrain.pth"), map_location=self.device))

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)
        logger.log("Pretraining policy")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                pred_actions, _ = dist.rsample()
                bc_loss = ((pred_actions - batch_act) ** 2).mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {i_epoch}, mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "rambo_pretrain.pth"))

    def update_dynamics(
        self, 
        real_buffer, 
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
            "adv_dynamics_update/adv_loss": 0, 
            "adv_dynamics_update/adv_advantage": 0, 
            "adv_dynamics_update/adv_log_prob": 0, 
        }
        self.dynamics.model.train() # danger

        steps = 0
        print("Start to train the dynamics model......")
        while steps < self._adv_train_steps:
            init_samples = real_buffer.sample_rollouts(self._adv_rollout_batch_size, self._adv_rollout_length)

            full_observations = init_samples["full_observations"][:, 0]
            full_actions = init_samples["full_actions"][:, 0]
            pre_actions = init_samples["pre_actions"][:, 0]
            time_steps = init_samples["time_steps"][:, 0]
            time_terminals = init_samples["terminals"][:, 0]

            idx_list = np.array(range(0, init_samples["full_observations"].shape[0]))

            if self._update_hidden_states:
                self.dynamics.reset(init_samples["hidden_states"]) 
            else:
                self.dynamics.reset()

            observations = full_observations[:, self.state_idxs]
            observations = self.sa_processor.get_rl_state(observations, init_samples["batch_idx_list"][0])
            tot_loss = 0.
            for t in range(self._adv_rollout_length):
                actions = super().select_action(observations) #??
                step_actions = self.sa_processor.get_step_action(actions)
                full_actions[:, self.action_idxs] = step_actions.copy()

                sl_input, sl_output, sl_mask = itemgetter("net_input", "net_output", "mask")(self.model_sl_buffer.sample(self._sl_batch_size))

                next_observations, terminals, loss_info, info, t_loss = self.dynamics_step_and_forward(observations, actions, full_observations, full_actions, 
                                                                                                       pre_actions, time_steps, time_terminals, sl_input, 
                                                                                                       sl_output, sl_mask, init_samples["batch_idx_list"][t],
                                                                                                       init_samples["batch_idx_list"][t+1])
                tot_loss += t_loss
                
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]

                steps += 1

                nonterm_mask = (~terminals[idx_list]).flatten()
                if nonterm_mask.sum() == 0 or steps >= self._adv_train_steps:
                    break
                
                observations = next_observations
                full_observations = info["next_full_observations"]
                time_steps = info["next_time_steps"]
                pre_actions = full_actions.copy()

                if t < self._adv_rollout_length - 1:
                    idx_list = idx_list[nonterm_mask]
                    full_actions = init_samples["full_actions"][:, t+1]
                    time_terminals = init_samples["terminals"][:, t+1]
            
            self._dynmics_adv_optim.zero_grad()
            tot_loss.backward() # TODO: update with sl_loss separately
            self._dynmics_adv_optim.step()

            # TODO: this may be too costly
            # print("Updating the hidden_states......")
            if self._update_hidden_states:
                real_buffer.update_hidden_states(self.dynamics.model, self.model_sl_buffer.net_input, self.model_sl_buffer.len_list)
                # real_buffer.update_hidden_states(self.dynamics.model)

        self.dynamics.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}

    def dynamics_step_and_forward(
        self,
        observations, 
        actions, 
        full_observations, 
        full_actions, 
        pre_actions, 
        time_steps,
        time_terminals, 
        sl_input, 
        sl_output, 
        sl_mask,
        batch_idxs,
        next_batch_idxs
    ):
        net_input = np.concatenate([full_observations, pre_actions, full_actions-pre_actions], axis=-1)
        mean, std = self.dynamics.model.forward(torch.tensor(net_input, device=self.device), is_tensor=True, with_grad=True)
        full_observations = torch.from_numpy(full_observations).to(self.device)
        mean += full_observations

        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.sample()
        ensemble_size, batch_size, output_size = ensemble_sample.shape

        # select the next observations
        info = {}
        selected_indexes = self.dynamics.model.random_member_idxs(batch_size)
        sample = ensemble_sample[selected_indexes, np.arange(batch_size)]
        full_next_observations = sample
        info["next_full_observations"] = sample.cpu().numpy()
        time_steps = time_steps + 1
        info["next_time_steps"] = time_steps.copy()

        # get the reward
        next_observations = full_next_observations[:, self.state_idxs]
        rewards = self.dynamics.reward_fn(next_observations.cpu().numpy(), batch_idxs).reshape(-1, 1)
        next_observations = self.sa_processor.get_rl_state(next_observations, next_batch_idxs)

        # get the termination signal
        terminals = self.dynamics.terminal_fn(time_steps)
        terminals = terminals | time_terminals

        # compute logprob, danger, TODO: select action_idxs
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        prob = log_prob.double().exp()
        prob = prob * (1.0/log_prob.shape[0])
        log_prob = prob.sum(0).log().type(torch.float32)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self._include_ent_in_adv:
                next_q = next_q - self._alpha * next_policy_log_prob

            value = torch.from_numpy(rewards).to(self.device) + (1-torch.from_numpy(terminals).to(self.device).float()) * self._gamma * next_q

            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std()+1e-6)
        adv_loss = (log_prob * advantage).mean()
    
        # compute the supervised loss, as used in the dynamics toolbox
        sl_loss = self.dynamics.model.get_sl_loss(sl_input, sl_output, sl_mask)    

        # sl_mean, sl_logvar = self.dynamics.model(sl_input)
        # sl_inv_var = torch.exp(-sl_logvar)
        # sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        # sl_var_loss = sl_logvar.mean(dim=(1, 2))
        # sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        # sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        # sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()

        all_loss = self._adv_weight * adv_loss + sl_loss
        # self._dynmics_adv_optim.zero_grad()
        # all_loss.backward()
        # self._dynmics_adv_optim.step() # TODO: do the step here

        return next_observations.cpu().numpy(), terminals, {
            "adv_dynamics_update/all_loss": all_loss.cpu().item(), 
            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(), 
            "adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(), 
            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item(), 
        }, info, all_loss

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return super().select_action(obs, deterministic)