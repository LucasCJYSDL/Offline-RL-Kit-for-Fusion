import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        
        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)
        
        self.full_observations = None
        self.full_next_observations = None
        self.pre_actions = None
        self.full_actions = None
        self.time_steps = None
        self.hidden_states = None
        self.net_input = None
        self.time_terminals = None
        self.batch_idx = None # used to query the tracking target at the current time step
        self.next_batch_idx = None 

    # may not be used
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    # fake buffer
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        full_obss: Optional[np.ndarray] = None,
        full_actions: Optional[np.ndarray] = None,
        pre_actions: Optional[np.ndarray] = None,
        time_steps: Optional[np.ndarray] = None,
        hidden_states: Optional[np.ndarray] = None,
        time_terminals: Optional[np.ndarray] = None,
        batch_idx: Optional[np.ndarray] = None,
        next_batch_idx: Optional[np.ndarray] = None
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        if full_obss is not None:
            # initialize if necessary
            if self.full_observations is None:
                self.full_observations = np.zeros((self._max_size,) + full_obss.shape[1:], dtype=full_obss.dtype)
                self.full_actions = np.zeros((self._max_size, full_actions.shape[-1]), dtype=full_actions.dtype)
                self.pre_actions = np.zeros((self._max_size, pre_actions.shape[-1]), dtype=pre_actions.dtype)
                self.time_steps = np.zeros((self._max_size, 1), dtype=np.float32)
                self.time_terminals = np.zeros((self._max_size, 1), dtype=np.float32)
                self.batch_idx = np.zeros((self._max_size, 1), dtype=np.int32)
                self.next_batch_idx = np.zeros((self._max_size, 1), dtype=np.int32)

            self.full_observations[indexes] = full_obss.copy()
            self.full_actions[indexes] = full_actions.copy()
            self.pre_actions[indexes] = pre_actions.copy()
            self.time_steps[indexes] = time_steps.copy()
            self.time_terminals[indexes] = time_terminals.copy()
            self.batch_idx[indexes] = batch_idx.copy()
            self.next_batch_idx[indexes] = next_batch_idx.copy()
        
        if hidden_states is not None:
            # initialize if necessary
            if self.hidden_states is None:
                self.hidden_states = np.zeros((self._max_size,) + hidden_states.shape[1:], dtype=hidden_states.dtype)
            
            self.hidden_states[indexes] = hidden_states.copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    # real buffer
    def load_dataset(self, dataset: Dict[str, np.ndarray], hidden=False) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)

        if "full_observations" in dataset:
            self.full_observations = np.array(dataset["full_observations"], dtype=self.obs_dtype)
            self.pre_actions = np.array(dataset["pre_actions"], dtype=self.action_dtype)
            self.full_actions = np.array(dataset["full_actions"], dtype=self.action_dtype)
            self.time_steps = np.array(dataset["time_step"]).reshape(-1, 1)
            if hidden:
                self.hidden_states = np.array(dataset["hidden_states"], dtype=np.float32)
                self.full_next_observations = np.array(dataset["full_next_observations"], dtype=self.obs_dtype)
    
    # real buffer
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    # real buffer and fake buffer
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        ret = {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }

        if self.full_observations is not None:
            ret["full_observations"] = torch.tensor(self.full_observations[batch_indexes]).to(self.device)
            ret["full_actions"] = torch.tensor(self.full_actions[batch_indexes]).to(self.device)
            ret["pre_actions"] = torch.tensor(self.pre_actions[batch_indexes]).to(self.device)
            ret["time_steps"] = torch.tensor(self.time_steps[batch_indexes]).to(self.device)

        if self.time_terminals is not None:
            ret["time_terminals"] = self.time_terminals[batch_indexes].astype(bool)
            ret["batch_idx"] = self.batch_idx[batch_indexes].copy()
            ret["next_batch_idx"] = self.next_batch_idx[batch_indexes].copy()
        
        if self.hidden_states is not None:
            # ret["hidden_states"] = torch.tensor(self.hidden_states[batch_indexes]).permute(1, 2, 0, 3).to(self.device)
            ret["hidden_states"] = torch.tensor(self.hidden_states[batch_indexes]).to(self.device)

        return ret

    # may not be used
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    
    # real buffer
    def sample_rollouts(self, batch_size, rollout_length) -> Dict[str, np.ndarray]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        f_obs, f_act, pre_act, time_step, terminal = [], [], [], [], []
        tmp_terminal = np.zeros_like(self.time_steps[batch_indexes])
        batch_idx_list = []

        for i in range(rollout_length+1):
            cur_batch_indexes = batch_indexes + i
            cur_batch_indexes[cur_batch_indexes >= self._size-1] = self._size-1
            batch_idx_list.append(cur_batch_indexes.copy())

            f_obs.append(self.full_observations[cur_batch_indexes].copy())
            f_act.append(self.full_actions[cur_batch_indexes].copy())
            pre_act.append(self.pre_actions[cur_batch_indexes].copy())
            time_step.append(self.time_steps[cur_batch_indexes].copy())
            if i >= 1:
                tmp_terminal[time_step[-1] <= time_step[-2]] = 1
                terminal.append(tmp_terminal.copy())

        ret = {
            "full_observations": np.moveaxis(np.array(f_obs)[:-1], source=0, destination=1),
            "full_actions": np.moveaxis(np.array(f_act)[:-1], source=0, destination=1),
            "pre_actions": np.moveaxis(np.array(pre_act)[:-1], source=0, destination=1),
            "time_steps": np.moveaxis(np.array(time_step)[:-1], source=0, destination=1),
            "terminals": np.moveaxis(np.array(terminal, dtype=bool), source=0, destination=1),
            "hidden_states": np.moveaxis(self.hidden_states[batch_indexes], source=0, destination=2),
            "batch_idx_list": np.array(batch_idx_list)
        }   
        
        return ret

    # real buffer
    def update_hidden_states(self, model, net_input, len_list):
        s_id = 0
        interval = 20
        size = net_input.shape[0]

        buffer_s_id = 0
        while s_id < size:
            e_id = min(s_id+interval, size)
            net_input_tensor = torch.FloatTensor(net_input[s_id:e_id]).to(model.device)

            memb_out_list = []
            with torch.no_grad():
                for memb in model.all_models:
                    net_input_n = memb.normalizer.normalize(net_input_tensor, 0)
                    memb_out = memb.get_mem_out(net_input_n).unsqueeze(1) # we have updated the dynamics toolbox
                    memb_out_list.append(memb_out)
            temp_hidden_state = torch.stack(memb_out_list, dim=1).permute(0, 3, 1, 2, 4).cpu().numpy()

            temp_hidden_state_list = []
            for i in range(e_id - s_id):
                temp_hidden_state_list.append(temp_hidden_state[i][:len_list[s_id+i]])
            valid_hidden_state = np.concatenate(temp_hidden_state_list, axis=0)
            buffer_e_id = buffer_s_id + valid_hidden_state.shape[0]
            self.hidden_states[buffer_s_id:buffer_e_id] = valid_hidden_state.copy()
            buffer_s_id = buffer_e_id

            s_id = e_id        
        


