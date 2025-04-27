"""
A template for a fusion env.
"""

import random
import numpy as np
import torch

from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir


class SA_processor: # used for both training and evaluation
    def __init__(self, offline_data, tracking_data, device):
        # for normalization or denormalization of the actuators
        bounds = (offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
        self.range = torch.FloatTensor((bounds[1] - bounds[0]) / 2.0).unsqueeze(0).to(device) # actuator bounds
        self.mid = torch.FloatTensor((bounds[1] + bounds[0]) / 2.0).unsqueeze(0).to(device)

        self.np_range = ((bounds[1] - bounds[0]) / 2.0)[np.newaxis, :]
        self.np_mid = ((bounds[1] + bounds[0]) / 2.0)[np.newaxis, :]

        # store the tracking taregts for both the training and evaluation data
        training_data_size = offline_data['tracking_ref'].shape[0]
        self.training_tracking_targets = np.array([offline_data['tracking_ref'][-1] for _ in range(training_data_size+1)])
        self.training_tracking_targets[:training_data_size] = offline_data['tracking_ref']

        self.eval_tracking_targets = {}
        self.eval_traj_states = {}
        self.eval_traj_actions = {}
        for key in tracking_data: # to avoid indexing issue, we use a padding trick here
            tracking_data_size = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_tracking_targets[key] = np.array([tracking_data[key]['tracking_ref'][-1] for _ in range(tracking_data_size+1)])
            self.eval_tracking_targets[key][:tracking_data_size] = tracking_data[key]['tracking_ref']
            # self.eval_data_sizes[key] = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_traj_states[key] = tracking_data[key]["tracking_states"]
            self.eval_traj_actions[key] = tracking_data[key]["tracking_actions"]
        
        # store idxs of the tracking targets to query them from the state
        # there could be multiple dimensions of the tracking target, so we may assign different coefficients to different dimensions.
        self.state_idxs = offline_data['state_idxs'] # selected state dimensions 
        self.action_idxs = offline_data["action_idxs"]
        self.idx_list = offline_data['index_list'] # indices corresponding to the target
        self.track_coefficients = np.array([1.0 for _ in range(len(self.idx_list))])
        coe = sum(self.track_coefficients)
        self.track_coefficients = self.track_coefficients / coe

        # get names of the tracking quantities and actuators in control
        self.tracking_target_names = offline_data['tracking_target_names']
        self.action_names = offline_data['action_names']
    
    def get_rl_state(self, state, batch_idx, shot_id=None):
        """
        Design of the state space (used for the rl policy).
        """
        is_np = True if type(state) == np.ndarray else False

        if np.isscalar(batch_idx):
            batch_idx = np.array([batch_idx for _ in range(state.shape[0])])
        else:
            batch_idx = np.array(batch_idx)

        if shot_id is None: # training data
            # batch_idx[batch_idx >= self.training_data_size] = self.training_data_size - 1
            targets = self.training_tracking_targets[batch_idx]
        else:
            # batch_idx[batch_idx >= self.eval_data_sizes[shot_id]] = self.eval_data_sizes[shot_id] - 1
            targets = self.eval_tracking_targets[shot_id][batch_idx]

        if not is_np:
            targets = torch.FloatTensor(targets).to(state.device)
        difference = targets - state[:, self.idx_list]

        if is_np:
            rl_state = np.concatenate([state, targets, difference], axis=-1) # the rl state contains the current state, tracking targets, and distance to the tracking targets
        else:
            rl_state = torch.cat([state, targets, difference], dim=-1)

        return rl_state
    
    def get_step_action(self, action):
        """
        The action from the rl model is from [-1, 1], we need to transform it back to the original range before inputing it to the dynamics model.
        """
        act_num = action.shape[0]
        if type(action) == np.ndarray:
            return action * np.repeat(self.np_range, repeats=act_num, axis=0) + np.repeat(self.np_mid, repeats=act_num, axis=0)
        
        return action * self.range.repeat(act_num, 1) + self.mid.repeat(act_num, 1)
    
    def normalize_action(self, action):
        """
        Normalize actions (based on the actuator bounds) in the offline dataset for rl training.
        """
        act_num = action.shape[0]
        return (action - self.mid.repeat(act_num, 1).cpu().numpy()) / (self.range.repeat(act_num, 1).cpu().numpy()+1e-6)
    
    def get_reward(self, next_state, time_step, shot_id=None): 
        """
        Design of the reward function.
        The reward function is defined based on the distance between the actual next state and the target next state (i.e., the target specified at the current time step).
        """
        if np.isscalar(time_step): # for evaluation
            assert shot_id is not None
            time_step = np.array([time_step])
            targets = self.eval_tracking_targets[shot_id][time_step]
        else:
            targets = self.training_tracking_targets[time_step] # for training
        # this design is flexible - we are using "-mse" as the reaward
        return -1.0 * (np.square(next_state[:, self.idx_list] - targets) * self.track_coefficients[np.newaxis, :]).sum(axis=1) 
    
    def get_plot_quantities(self, shot_id, time_step, state, action):
        """
        According to the shot id and time step, query the quantity target, real quantity, achieved quantity, real actuators, and actuators adopted by the controller.
        """
        target_quan = self.eval_tracking_targets[shot_id][time_step]
        real_quan = self.eval_traj_states[shot_id][time_step][self.state_idxs][self.idx_list]
        cur_quan = state[0, self.idx_list].cpu().numpy() # Note that this is correct only because state is the first component of the rl_state. 
        real_act = self.eval_traj_actions[shot_id][time_step][self.action_idxs]
        cur_act = self.get_step_action(action)[0]

        return target_quan, real_quan, cur_quan, real_act, cur_act
    
    def get_plot_names(self):
        """
        Return the list of tracking targets and controllable actuators.
        """
        return self.tracking_target_names, self.action_names


class NFBaseEnv: # env for evaluation
    def __init__(self, model_dir, sa_processor, general_data, tracking_data, ref_shot_id, device):
        state_idxs, action_idxs = general_data['state_idxs'], general_data['action_idxs']
        tracking_states, tracking_pre_actions, tracking_actions = tracking_data['tracking_states'], tracking_data['tracking_pre_actions'], \
                                                                  tracking_data['tracking_actions']
        # load the variables
        self.cur_time = None
        self.ref_shot_id = ref_shot_id
        self.time_limit = tracking_states.shape[0] # this should be the horizon of the reference shot
        self.tracking_states = np.array(tracking_states)
        self.tracking_pre_actions = np.array(tracking_pre_actions)
        self.tracking_actions = np.array(tracking_actions)

        self.device = device
        self.sa_processor = sa_processor

        self.state_idxs = state_idxs
        self.action_idxs = action_idxs

        # load the well-trained rnn model ensemble, which is the backbone of this fusion env.
        ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir) # TODO: an ensemble of dynamics models
        self.all_models = ensemble.members
        for memb in self.all_models:
            memb.to(device)
            memb.eval()
    
    def get_eval_shot_list(self):
        """
        return the list of shots for evaluation
        """
        return [self.ref_shot_id]

    def seed(self, seed):
        """
        Seed the randomness.
        """
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, shot_id=None):
        """
        Reset at the beginning of an episode.
        """
        self.cur_time = random.randint(0, 9)
        self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)

        # reset the model
        for memb in self.all_models:
            memb.reset()
        
        return_state = self.cur_state[:, self.state_idxs]
        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id = self.ref_shot_id)

    def step(self, cur_action):
        """
        Proceed to the next time step in the episode.
        """
        # prepare the input for the dymamics model
        cur_action = torch.Tensor(cur_action).to(self.device)
        cur_action = self.sa_processor.get_step_action(cur_action)
        cur_action_pad = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).to(self.device)
        cur_action_pad[:, self.action_idxs] = cur_action
        cur_action = cur_action_pad

        net_input = torch.cat([self.cur_state, self.pre_action, cur_action-self.pre_action], dim=-1)

        # get the ensemble output
        ensemble_preds = 0.
        with torch.no_grad():
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, _ = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                net_output = memb.normalizer.unnormalize(net_output_n, 1)
                ensemble_preds += net_output
        ensemble_preds = ensemble_preds / float(len(self.all_models)) # delta of the state, which is the mean of the ensemble outputs

        # proceed to the next time step
        self.cur_state = self.cur_state + ensemble_preds # the next state, TODO: use the true value for the unselected dimensions
        return_state = self.cur_state[:, self.state_idxs]
        reward = self.get_reward(return_state.cpu().numpy(), self.cur_time, shot_id=self.ref_shot_id)[0] # next state and current time step
        self.cur_time += 1
        self.pre_action = cur_action.clone()

        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id=self.ref_shot_id), reward, self.is_done(self.cur_time), {"time_step": self.cur_time}

    def get_reward(self, next_state, time_step, shot_id):
        return self.sa_processor.get_reward(next_state, time_step, shot_id)
        
    def is_done(self, time_step):
        # terminates when exceeding the time limit of the shot
        return time_step >= self.time_limit
