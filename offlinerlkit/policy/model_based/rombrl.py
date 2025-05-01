import numpy as np
import torch
import torch.nn as nn
from operator import itemgetter
from typing import Dict, Union, Tuple
from tqdm import tqdm

from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy import MOBILEPolicy
from offlinerlkit.buffer import RobustRolloutBuffer, ModelSLReplayBuffer


class ROMBRLPolicy(MOBILEPolicy):
    """
    ROMBRL: Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning
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
        max_q_backup: bool = False,
        # new
        small_traj_batch: bool = False,
        dynamics_adv_optim: torch.optim.Optimizer = None,
        model_sl_buffer: ModelSLReplayBuffer = None,
        onpolicy_buffer: RobustRolloutBuffer = None,
        grad_mode: int = 3,
        I_coe: float = 0.0,
        epsilon: float = 10.0,
        down_sample_size: int = 8,
        sl_weight: float = 0,
        lambda_training_epoch: int = 1,
        lambda_lr: float = 1e-3,
        onpolicy_rollout_length: int = 5,
        onpolicy_rollout_batch_size: int = 2500,
        onpolicy_batch_size: int = 256,
        clip_range: float = 0.2,
        actor_training_epoch: int = 10,
        dynamics_training_epoch: int = 10,
        include_ent_in_adv: bool = False,
        update_hidden_states: bool = False,
        scaler: StandardScaler = None, 
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics,
            actor,
            critics,
            actor_optim,
            critics_optim,
            state_idxs,
            action_idxs,
            sa_processor,
            tau,
            gamma,
            alpha,
            penalty_coef,
            num_samples,
            deterministic_backup,
            max_q_backup
        )
        # dynamics update
        self.model_sl_buffer = model_sl_buffer
        self.dynmics_adv_optim = dynamics_adv_optim
        self.dynamics_training_epoch = dynamics_training_epoch
        self.include_ent_in_adv = include_ent_in_adv
        self._update_hidden_states = update_hidden_states
        
        # actor update
        self.onpolicy_rollout_length = onpolicy_rollout_length
        self.onpolicy_rollout_batch_size = onpolicy_rollout_batch_size
        self.onpolicy_batch_size = onpolicy_batch_size
        self.sl_batch_size = onpolicy_batch_size // 16 # TODO: hacky
        self.actor_training_epoch = actor_training_epoch
        self.onpolicy_buffer = onpolicy_buffer
        self.clip_range = clip_range
        self._epsilon = epsilon
        self.down_sample_size = down_sample_size
        self.small_traj_batch = small_traj_batch
        self._I_coe = I_coe
        
        # dual variable update
        self._lambda = sl_weight
        self.lambda_training_epoch = lambda_training_epoch
        self.lambda_lr = lambda_lr

        # basic setting
        self._grad_mode = grad_mode
        self.scaler = scaler
        self.device = device
    
    def train(self) -> None:
        self.actor.train()
        self.critics.train()
        self.dynamics.model.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()
        self.dynamics.model.eval()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return super().select_action(obs, deterministic)
    
    def learn(self, batch: Dict) -> Dict[str, float]: 
        # go through the dynamics model again, since the underlying dynamics model is also being updated
        # TODO: skip this
        if self._update_hidden_states:
            self.dynamics.reset(batch["fake"]["hidden_states"].permute(1, 2, 0, 3))
        else:
            self.dynamics.reset()
        next_obss, rewards, terminals, info = self.dynamics.step(batch["fake"]["full_observations"].cpu().numpy(), batch["fake"]["pre_actions"].cpu().numpy(), batch["fake"]["full_actions"].cpu().numpy(),
                                                                 batch["fake"]["time_steps"].cpu().numpy(), batch["fake"]["time_terminals"], self.state_idxs, batch["fake"]["batch_idx"].flatten())
        next_obss = self.sa_processor.get_rl_state(next_obss, batch["fake"]["next_batch_idx"].flatten())
        batch["fake"]["next_observations"] = torch.FloatTensor(next_obss).to(self.device)
        batch["fake"]["rewards"] = torch.FloatTensor(rewards).to(self.device)
        batch["fake"]["terminals"] = torch.FloatTensor(terminals).to(self.device)
        
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
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
    
    def _get_qvs(self, obs, act):
        obs = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            q_values = torch.min(torch.cat([critic(obs, act) for critic in self.critics], 1), 1)[0].reshape(-1, 1)
       
            new_act, _ = self.actforward(obs, deterministic=False)
            values = torch.min(torch.cat([critic(obs, new_act) for critic in self.critics], 1), 1)[0].reshape(-1, 1)
        
        return q_values, values
        
    def _update_init_samples(self, init_samples, new_samples, terminals, starting_idx):
        for key in init_samples:
            if key in ["hidden_states", "batch_idx_list"]:
                continue
            padding_length = init_samples[key].shape[1] - starting_idx
            init_samples[key][terminals, starting_idx:] = new_samples[key][:, :padding_length]
        # a little bit weird 
        padding_length = init_samples["batch_idx_list"].shape[0] - starting_idx
        init_samples["batch_idx_list"][starting_idx:, terminals] = new_samples["batch_idx_list"][:padding_length]

    def _reset_dynamics_mem(self, new_samples, terminals):
        hidden_states = np.moveaxis(self.dynamics.get_memory(), source=0, destination=2)
        if self._update_hidden_states:
            hidden_states[:, :, terminals] = 0.0
        else:
            hidden_states[:, :, terminals] = new_samples["hidden_states"]
        self.dynamics.reset(hidden_states)

    def _onpolicy_rollout(self, real_buffer):
        self.onpolicy_buffer.reset()

        init_samples = real_buffer.sample_rollouts(self.onpolicy_rollout_batch_size, self.onpolicy_rollout_length)
        full_observations = init_samples["full_observations"][:, 0]
        full_actions = init_samples["full_actions"][:, 0]
        pre_actions = init_samples["pre_actions"][:, 0]
        time_steps = init_samples["time_steps"][:, 0]
        time_terminals = init_samples["terminals"][:, 0]

        if self._update_hidden_states:
            self.dynamics.reset(init_samples["hidden_states"]) 
        else:
            self.dynamics.reset()

        # rollout
        observations = full_observations[:, self.state_idxs]
        observations = self.sa_processor.get_rl_state(observations, init_samples["batch_idx_list"][0])
        with torch.no_grad():
            actions, log_probs = self.actforward(observations, deterministic=False)
        last_terminals = np.zeros((observations.shape[0], ), dtype=bool)

        for t in range(self.onpolicy_rollout_length):
            q_values, values = self._get_qvs(observations, actions)
            
            # prepare for the step function
            actions = actions.cpu().numpy()
            step_actions = self.sa_processor.get_step_action(actions)
            full_actions[:, self.action_idxs] = step_actions.copy()

            # hidden_states = self.dynamics.get_memory()
            next_observations, rewards, terminals, info = self.dynamics.step(full_observations, pre_actions, full_actions, time_steps, time_terminals, self.state_idxs, init_samples["batch_idx_list"][t])
            next_observations = self.sa_processor.get_rl_state(next_observations, init_samples["batch_idx_list"][t+1])

            log_probs = log_probs.flatten()
            dyn_net_input = info['dyn_net_input']
            log_dyn_probs = info['log_dyn_probs'].flatten()
            dyn_samples = info['dyn_samples']
            dyn_model_idxs = info['model_idxs']
            rewards = rewards.flatten()
            terminals = terminals.flatten()
            num_terminals = terminals.sum()

            # self.onpolicy_buffer.add(observations, actions, rewards, last_terminals, values, log_probs, q_values, dyn_net_input, hidden_states, log_dyn_probs, dyn_samples, dyn_model_idxs)
            self.onpolicy_buffer.add(observations, actions, rewards, last_terminals, values, log_probs, q_values, dyn_net_input, log_dyn_probs, dyn_samples, dyn_model_idxs)

            # prepare for the next time step
            full_observations = info["next_full_observations"]
            if t < self.onpolicy_rollout_length - 1:
                time_steps = info["next_time_steps"]
                pre_actions = full_actions.copy()
                full_actions = init_samples["full_actions"][:, t+1]
                time_terminals = init_samples["terminals"][:, t+1]
            
            # if some envs terminate, we need to reset them.
            if num_terminals > 0:
                new_samples = real_buffer.sample_rollouts(num_terminals, self.onpolicy_rollout_length - t) # this rollout length is an important parameter
                full_observations[terminals] = new_samples["full_observations"][:, 0]

                new_next_observations = full_observations[terminals][:, self.state_idxs]
                new_next_observations = self.sa_processor.get_rl_state(new_next_observations, new_samples["batch_idx_list"][0])
                next_observations[terminals] = new_next_observations

                if t < self.onpolicy_rollout_length - 1:
                    time_steps[terminals] = new_samples["time_steps"][:, 0]
                    pre_actions[terminals] = new_samples["pre_actions"][:, 0]
                    full_actions[terminals] = new_samples["full_actions"][:, 0]
                    time_terminals[terminals] = new_samples["terminals"][:, 0]
                    self._update_init_samples(init_samples, new_samples, terminals, t+1)
                    self._reset_dynamics_mem(new_samples, terminals) # danger

            observations = next_observations
            with torch.no_grad():
                actions, log_probs = self.actforward(observations, deterministic=False)
            last_terminals = terminals
        
        # after rollout
        with torch.no_grad():
            # compute value for the last timestep
            q_values, values = self._get_qvs(observations, actions)  # type: ignore[arg-type]

        self.onpolicy_buffer.compute_returns_and_advantage(last_values=values, last_q_values=q_values, dones=terminals)
        # self.onpolicy_buffer.split_into_episodes()
    
    def _evaluate_dynamics(self, dyn_input, dyn_sample, input_mask, dyn_model_idxs):
        # preparation
        input_mask = input_mask.reshape(-1)
        dyn_sample = dyn_sample.reshape(-1, dyn_sample.shape[-1])[input_mask > 0]
        dyn_sample = torch.tensor(dyn_sample, device=self.device)
        
        # inference
        self.dynamics.reset() # actually, we should reset with hidden states when inference with rollout data
        mean, std = self.dynamics.model.get_net_out(dyn_input, input_mask) 
        
        if dyn_model_idxs is None:
            dyn_model_idxs = self.dynamics.model.random_member_idxs(mean.shape[1])
        else:
            dyn_model_idxs = dyn_model_idxs.reshape(-1)[input_mask > 0]
            dyn_model_idxs = torch.tensor(dyn_model_idxs, device=self.device)

        mean = mean[dyn_model_idxs, np.arange(dyn_model_idxs.shape[0])]
        std = std[dyn_model_idxs, np.arange(dyn_model_idxs.shape[0])]

        dist = torch.distributions.Normal(mean, std) # TODO: logvar has been clamped

        return dist.log_prob(dyn_sample).sum(dim=-1)

    def _get_discount_tensor(self, episode_starts: np.ndarray):
        seq_len, batch_size = episode_starts.shape[0], episode_starts.shape[1]
        discount_tensor = []
        for b in range(batch_size):
            tmp_time = 0
            discount_list = []
            for i in range(seq_len):
                if episode_starts[i][b] > 0:
                    tmp_time = 0
                discount_list.append(self._gamma ** tmp_time)
                tmp_time += 1
            discount_tensor.append(discount_list)
        
        return torch.tensor(np.array(discount_tensor), dtype=torch.float32, device=self.device).permute(1, 0)

    def _evaluate_rollout_dynamics(self, dyn_inputs: np.ndarray, episode_starts: np.ndarray, dyn_samples: np.ndarray, dyn_model_idxs: np.ndarray):
        # get input episodes and masks
        seq_len, batch_size, input_size = dyn_inputs.shape
        _, _, output_size = dyn_samples.shape

        episodes_inputs = []
        episodes_samples = []
        episodes_model_idxs = []
        episodes_masks = []

        for b in range(batch_size):
            start_indices = np.where(episode_starts[:, b])[0]
            start_indices = np.insert(start_indices, 0, 0)  # add 0 at the beginning
            start_indices = np.append(start_indices, seq_len)  # add end of sequence as the last index
            
            for i in range(len(start_indices) - 1):
                start, end = start_indices[i], start_indices[i + 1]
                episode_input = dyn_inputs[start:end, b]
                episode_sample = dyn_samples[start:end, b]
                episode_model_idx = dyn_model_idxs[start:end, b]

                episodes_inputs.append(episode_input)
                episodes_samples.append(episode_sample)
                episodes_model_idxs.append(episode_model_idx)
                episodes_masks.append(np.ones((end - start, 1), dtype=np.float32))  # mask for valid entries

        # find the maximum episode length for padding
        max_episode_len = max(ep.shape[0] for ep in episodes_inputs)

        # pad episodes to the maximum length
        padded_inputs = np.zeros((len(episodes_inputs), max_episode_len, input_size), dtype=np.float32)
        padded_samples = np.zeros((len(episodes_samples), max_episode_len, output_size), dtype=np.float32)
        padded_model_idxs = np.zeros((len(episodes_model_idxs), max_episode_len), dtype=np.int32)
        padded_masks = np.zeros((len(episodes_masks), max_episode_len, 1), dtype=np.float32)

        for i, (ep_input, ep_sample, ep_model_idx, ep_mask) in enumerate(zip(episodes_inputs, episodes_samples, episodes_model_idxs, episodes_masks)):
            padded_inputs[i, :ep_input.shape[0]] = ep_input
            padded_samples[i, :ep_sample.shape[0]] = ep_sample
            padded_model_idxs[i, :ep_model_idx.shape[0]] = ep_model_idx
            padded_masks[i, :ep_mask.shape[0]] = ep_mask

        # evaluate dynamics
        masked_log_dyn_prob = self._evaluate_dynamics(padded_inputs, padded_samples, padded_masks, padded_model_idxs)
        
        return masked_log_dyn_prob.reshape(batch_size, seq_len).permute(1, 0) # danger

    def _get_sl_loss(self, sl_observations, sl_actions, sl_next_observations, sl_rewards):
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1)
        sl_target = torch.cat([sl_next_observations-sl_observations, sl_rewards], dim=-1)
        sl_input = self.dynamics.scaler.transform_tensor(sl_input)

        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2)) # TODO: using log likelihood
        sl_var_loss = sl_logvar.mean(dim=(1, 2))

        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()

        return sl_loss

    def _get_grad_list(self, log_prob_list, model):
        # TODO: the time cost here can be improved, as a trade off of the memory cost
        grad_list = []
        for t in range(log_prob_list.shape[0]):
            model.zero_grad()
            if t < log_prob_list.shape[0] - 1:
                log_prob_list[t].backward(retain_graph=True)
            else:
                log_prob_list[t].backward()
            grad_list.append(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).cpu().numpy())
        
        return grad_list
    
    def _apply_grad(self, grad_vector):
        grad_vector = torch.FloatTensor(grad_vector).to(self.device)

        self.actor.zero_grad() # TODO: optional
        offset = 0
        for param in self.actor.parameters():
            if param.requires_grad:
                numel = param.numel()
                param.grad = grad_vector[offset:offset + numel].view_as(param)  # assign custom gradient
                offset += numel
        
        self.actor_optim.step()
    
    def update_dynamics(
        self, 
        real_buffer
    ): # hacky, it's actually updating the actor, dynamics model, and Lagrangian multiplier (optional)
        self._onpolicy_rollout(real_buffer)

        # update lambda
        # print("Updating the Lagrandian multiplier ......") #??
        # lambda_grad_list = []
        # temp_lambda = self._lambda
        # for _ in range(self.lambda_training_epoch):
        #     sl_input, sl_output, sl_mask = itemgetter("net_input", "net_output", "mask")(self.model_sl_buffer.sample(self.sl_batch_size)) # TODO: use :10
        #     with torch.no_grad():
        #         real_log_dyn_prob = self._evaluate_dynamics(sl_input, sl_output, sl_mask, dyn_model_idxs=None)

        #     C = (- real_log_dyn_prob.mean() - self._epsilon).cpu().numpy()
        #     # TODO: using phi_bar, Adam
        #     temp_lambda = temp_lambda + self.lambda_lr * C
        #     temp_lambda = max(temp_lambda, 1e-3) # danger
        #     lambda_grad_list.append(self.lambda_lr * C)
        
        # update the actor
        print("Updating the actor ......")
        gradient_mask_ratio_list, J_list, J_phi_list, nabla_theta_J_norm_list, H_nabla_phi_J_norm_list, Delta_theta_J_list = [], [], [], [], [], []
        for epoch_id in tqdm(range(self.actor_training_epoch)):

            if self.small_traj_batch:
                traj_batch_size = self.onpolicy_batch_size // 10
            else:
                traj_batch_size = self.onpolicy_batch_size

            for rollout_traj in self.onpolicy_buffer.get_traj(traj_batch_size): 
                # get common tensors
                discount_tensor = self._get_discount_tensor(rollout_traj.episode_starts)
                # discount_tensor = torch.FloatTensor([[self._gamma ** i] for i in range(self.onpolicy_rollout_length)]).to(self.device)

                # normalize advantages
                advantages = rollout_traj.advantages
                advantages = (advantages - advantages.mean(dim=-1, keepdim=True)) / (advantages.std(dim=-1, keepdim=True) + 1e-8) # why dim = -1?

                # get gradient masks for the actor
                dist = self.actor(rollout_traj.observations)
                log_prob = dist.log_prob(rollout_traj.actions).squeeze(dim=-1) # TODO: train with the actor's entropy
                ratio = torch.exp(log_prob - rollout_traj.old_log_prob)
                adv_loss_1 = advantages * ratio
                adv_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                gradient_masks = (adv_loss_1 <= adv_loss_2).to(torch.float32).detach()
                
                # get the first gradient term 
                J = gradient_masks * advantages * log_prob * discount_tensor # TODO: using returns (defined by rewards) instead of advantages
                J = (J.sum(dim=0)).mean()

                # J = torch.min(adv_loss_1, adv_loss_2).mean() # PPO loss
                
                self.actor.zero_grad() # danger
                J.backward(retain_graph=True)
                nabla_theta_J = torch.cat([p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None]).cpu().numpy() # torch.Size([73484])

                # NOTE: if the dynamics model is large, it's not plausible to run grad_mode 2 or 3.
                if self._grad_mode == 1 or epoch_id % 2:
                    self._apply_grad(-nabla_theta_J) # Eq. (7)
                    # loggings
                    gradient_mask_ratio_list.append(gradient_masks.mean().item())
                    J_list.append(J.item())
                    continue

                # get W
                cur_batch_size = rollout_traj.observations.shape[1]
                down_sample_indices = torch.randperm(cur_batch_size)[:self.down_sample_size] # get down samples, it would be time costly is using the full batch
                UV_denominator = np.sqrt(self.down_sample_size) # sqrt(m)
                sampled_log_prob = (gradient_masks[:, down_sample_indices] * log_prob[:, down_sample_indices]).sum(0)

                self.W = np.array(self._get_grad_list(sampled_log_prob, self.actor)) / UV_denominator

                # get nabla_phi_J
                # you can choose to use a shorter length (10) or a smaller batch size to save gpu memory
                log_dyn_prob = self._evaluate_rollout_dynamics(rollout_traj.dyn_net_inputs[:10], rollout_traj.episode_starts[:10], 
                                                               rollout_traj.dyn_samples[:10], rollout_traj.dyn_model_idxs[:10]) 

                dyn_advantages = rollout_traj.dyn_advantages[:10] # normalize dyn_advantages
                dyn_advantages = (dyn_advantages - dyn_advantages.mean(dim=-1, keepdim=True)) / (dyn_advantages.std(dim=-1, keepdim=True) + 1e-8) # why dim = -1?
                discount_tensor = discount_tensor[:10]
                
                J_phi = dyn_advantages * log_dyn_prob * discount_tensor
                J_phi = (J_phi.sum(dim=0)).mean()

                self.dynamics.model.zero_grad() # danger
                J_phi.backward(retain_graph=True)
                nabla_phi_J = torch.cat([p.grad.view(-1) for p in self.dynamics.model.parameters() if p.grad is not None]).unsqueeze(-1).cpu().numpy() # torch.Size([928488])
                dyn_advantages = dyn_advantages.cpu().numpy()
                discount_tensor = discount_tensor.cpu().numpy()
                
                # get (U, V), (X, Y)
                sampled_log_dyn_prob = log_dyn_prob[:, down_sample_indices].reshape(-1)

                self.V = np.array(self._get_grad_list(sampled_log_dyn_prob, self.dynamics.model)).reshape(discount_tensor.shape[0], self.down_sample_size, -1)
                self.U = (dyn_advantages[:, down_sample_indices, np.newaxis] * self.V * discount_tensor[:, down_sample_indices, np.newaxis])

                ## get (X, Y)
                uniform_indices = np.random.choice(discount_tensor.shape[0], size=self.down_sample_size, replace=True)
                self.X = self.U[uniform_indices, np.arange(self.down_sample_size)] / UV_denominator # M/h = m
                self.Y = self.V[uniform_indices, np.arange(self.down_sample_size)] / UV_denominator # TODO: the samples for estimating (U, V) and (X, Y) can be different
                
                self.U = self.U.sum(0) / UV_denominator
                self.V = self.V.sum(0) / UV_denominator

                # get C (and B), TODO: using phi_bar
                sl_input, sl_output, sl_mask = itemgetter("net_input", "net_output", "mask")(self.model_sl_buffer.sample(self.sl_batch_size)) # TODO: use :10
                real_log_dyn_prob = self._evaluate_dynamics(sl_input, sl_output, sl_mask, dyn_model_idxs=None)

                if self._grad_mode == 3:
                    C = - real_log_dyn_prob.mean() - self._epsilon 
                    self.dynamics.model.zero_grad() # danger
                    C.backward(retain_graph=True)
                    B = torch.cat([p.grad.view(-1) for p in self.dynamics.model.parameters() if p.grad is not None]).unsqueeze(-1).cpu().numpy()
                    C = C.detach().cpu().numpy()

                # get Z
                down_sample_indices = torch.randperm(real_log_dyn_prob.shape[0])[:(self.down_sample_size * discount_tensor.shape[0])] # TODO: do not use self.onpolicy_rollout_length
                sampled_real_log_dyn_prob = real_log_dyn_prob[down_sample_indices]

                self.Z = np.array(self._get_grad_list(sampled_real_log_dyn_prob, self.dynamics.model)) * np.sqrt(self._lambda / self.down_sample_size / discount_tensor.shape[0]) # TODO: learn the lambda
                print(self.Z.shape)
                # get caching values necessary for inverse calculation
                self.A_cache, self.M2_cache, self.M3_cache = None, None, None
                self._caching_values()

                # Eq. (8)
                if self._grad_mode == 2:
                    H_nabla_phi_J = self._A_inv(nabla_phi_J) 
                # Eq. (9)
                else:
                    # get S
                    A_inv_B = self._A_inv(B)
                    S = C - self._lambda * (B.T @ A_inv_B)[0, 0]

                    # get H * nabla_phi_J
                    A_inv_nabla_phi_J = self._A_inv(nabla_phi_J)
                    temp_P = B.T @ A_inv_nabla_phi_J
                    H_nabla_phi_J = A_inv_nabla_phi_J + self._lambda * A_inv_B @ temp_P / S
                
                # get and update with the total derivative
                temp_M = self.U @ H_nabla_phi_J
                Delta_theta_J = nabla_theta_J - 1e-8 * (self.W.T @ temp_M)[:, 0] # TODO: apply a weight on the second term
                self._apply_grad(-Delta_theta_J) # danger

                # loggings
                gradient_mask_ratio_list.append(gradient_masks.mean().item())
                J_list.append(J.item())
                J_phi_list.append(J_phi.item())
                nabla_theta_J_norm_list.append(np.linalg.norm(nabla_theta_J))
                H_nabla_phi_J_norm_list.append(np.linalg.norm(H_nabla_phi_J))
                Delta_theta_J_list.append(np.linalg.norm(Delta_theta_J))


        # release the memory
        self.W, self.U, self.V, self.X, self.Y, self.Z = None, None, None, None, None, None
        self.A_cache, self.M2_cache, self.M3_cache = None, None, None

        # update the dynamics
        print("Updating the world model ......")
        dyn_adv_loss_list, sl_loss_list, dyn_all_loss_list = [], [], []
        for _ in tqdm(range(self.dynamics_training_epoch)):
            for rollout_data in self.onpolicy_buffer.get_traj(self.sl_batch_size): # TODO: maybe too much
                log_dyn_prob = self._evaluate_rollout_dynamics(rollout_data.dyn_net_inputs, rollout_data.episode_starts, 
                                                               rollout_data.dyn_samples, rollout_data.dyn_model_idxs)
                
                # normalize dyn advantage
                dyn_advantages = rollout_data.dyn_advantages
                dyn_advantages = (dyn_advantages - dyn_advantages.mean()) / (dyn_advantages.std() + 1e-8)

                # ratio between old and new dyn, should be one at the first iteration
                ratio = torch.exp(log_dyn_prob - rollout_data.old_log_dyn_prob)

                # clipped surrogate loss
                dyn_adv_loss_1 = dyn_advantages * ratio
                dyn_adv_loss_2 = dyn_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                dyn_adv_loss = torch.min(dyn_adv_loss_1, dyn_adv_loss_2).mean() # TODO: using log pi * adv instead

                # dyn_gradient_masks = (dyn_adv_loss_1 <= dyn_adv_loss_2).to(torch.float32).detach()
                # dyn_adv_loss = (dyn_gradient_masks * dyn_advantages * log_dyn_prob).mean() # TODO: using returns (defined by rewards) instead of advantages

                # get samples for sl, TODO: using \phi_\bar
                sl_input, sl_output, sl_mask = itemgetter("net_input", "net_output", "mask")(self.model_sl_buffer.sample(self.sl_batch_size))

                # compute the supervised loss
                sl_loss = self.dynamics.model.get_sl_loss(sl_input, sl_output, sl_mask)    

                # step with the overall loss
                dyn_all_loss = dyn_adv_loss / self._lambda + sl_loss
                self.dynmics_adv_optim.zero_grad()
                dyn_all_loss.backward() # TODO: apply gradient clips

                torch.nn.utils.clip_grad_norm_(self.dynamics.model.parameters(), 1.0) # TODO: fine-tune this grad norm
                self.dynmics_adv_optim.step()

                dyn_adv_loss_list.append(dyn_adv_loss.item())
                sl_loss_list.append(sl_loss.item()) 
                dyn_all_loss_list.append(dyn_all_loss.item())

        # since the dynamics model has been updated, we should also update the hidden states.
        if self._update_hidden_states:
            real_buffer.update_hidden_states(self.dynamics.model, self.model_sl_buffer.net_input, self.model_sl_buffer.len_list)
            # the hidden states in the fake buffer should also be updated, but it's too costly

        # lambda_k -> lambda_{k+1}
        # self._lambda = temp_lambda # TODO: relocate this #??

        # final loggings
        all_loss_info = {
            "dyn_update/dyn_all_loss": np.mean(dyn_all_loss_list), 
            "dyn_update/sl_loss": np.mean(sl_loss_list), 
            "dyn_update/dyn_adv_loss": np.mean(dyn_adv_loss_list),

            # "lambda_update/lambda": self._lambda, #??
            # "lambda_update/gradient": np.mean(lambda_grad_list),

            "actor_update/grad_mask_ratio": np.mean(gradient_mask_ratio_list),
            "actor_update/adv_loss": np.mean(J_list),
        }
        if self._grad_mode > 1:
            all_loss_info["actor_update/dyn_adv_loss"] = np.mean(J_phi_list)
            all_loss_info["actor_update/J_norm"] = np.mean(nabla_theta_J_norm_list)
            all_loss_info["actor_update/H_norm"] = np.mean(H_nabla_phi_J_norm_list)
            all_loss_info["actor_update/tot_dev_norm"] = np.mean(Delta_theta_J_list)

        return all_loss_info

    def _caching_values(self):
        # get M2_cache
        temp_P = self._M3_inv(self.Z.T)
        self.M2_cache = temp_P @ np.linalg.inv(np.eye(self.Z.shape[0]) + self.Z @ temp_P)

        # get A_cache
        temp_M = self._M2_inv(self.U.T)
        self.A_cache = temp_M @ np.linalg.inv(np.eye(self.V.shape[0]) + self.V @ temp_M)

    def _A_inv(self, P):
        temp_P = self._M2_inv(P)
        temp_M = self.V @ temp_P

        return temp_P - self.A_cache @ temp_M

    def _M2_inv(self, P):
        temp_P = self._M3_inv(P)
        temp_M = self.Z @ temp_P

        return temp_P - self.M2_cache @ temp_M
    
    def _M3_inv(self, P):
        if self.M3_cache is None:
            # get M3_cache
            temp_P = self.Y @ self.X.T
            temp_P = np.linalg.inv(self._I_coe * np.eye(self.Y.shape[0]) - temp_P)
            self.M3_cache = self.X.T @ temp_P
        
        temp_M = self.Y @ P
        return (P + self.M3_cache @ temp_M) / self._I_coe