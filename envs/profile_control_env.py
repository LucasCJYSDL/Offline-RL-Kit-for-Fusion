import torch
import random
import numpy as np

from envs.base_env import NFBaseEnv


class ProfileControlEnv(NFBaseEnv): # env for evaluation
    def __init__(self, model_dir, sa_processor, general_data, tracking_data, ref_shot_id, device):
        super().__init__(model_dir, sa_processor, general_data, tracking_data[ref_shot_id], ref_shot_id, device)
        # these variables are from the base env but we don't need them
        self.ref_shot_id = None
        self.tracking_states, self.tracking_pre_actions, self.tracking_actions = None, None, None
        self.eval_shot_list = list(tracking_data.keys())
        self.tracking_data = tracking_data

    def get_eval_shot_list(self):
        """
        return the list of shots for evaluation
        """
        return self.eval_shot_list

    def reset(self, shot_id=None):
        # randomly sample a shot for evaluation
        if shot_id is None:
            self.ref_shot_id = random.choice(self.eval_shot_list)
        else:
            self.ref_shot_id = shot_id

        self.tracking_states, self.tracking_pre_actions, self.tracking_actions = self.tracking_data[self.ref_shot_id]['tracking_states'], \
                                                                                 self.tracking_data[self.ref_shot_id]['tracking_pre_actions'], \
                                                                                 self.tracking_data[self.ref_shot_id]['tracking_actions']
        self.cur_shot_time_limit = self.tracking_states.shape[0]
        
        # randomly sample an initial time step
        # self.cur_time = random.randint(0, 9) # TODO
        self.cur_time = 0
        self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)

        # reset the model
        for memb in self.all_models:
            memb.reset()
        
        return_state = self.cur_state[:, self.state_idxs]
        
        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id=self.ref_shot_id)

    def step(self, cur_action):
        # prepare the input for the dymamics model
        cur_action = torch.Tensor(cur_action).to(self.device)
        batch_size = cur_action.shape[0] # step with a batch of actions
        cur_action = self.sa_processor.get_step_action(cur_action)
        cur_action_pad = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        cur_action_pad[:, self.action_idxs] = cur_action
        cur_action = cur_action_pad

        if self.cur_state.shape[0] < batch_size:
            self.cur_state = self.cur_state.repeat(batch_size, 1)
            self.pre_action = self.pre_action.repeat(batch_size, 1)
        net_input = torch.cat([self.cur_state, self.pre_action, cur_action-self.pre_action], dim=-1)

        # get the ensemble output
        ensemble_preds = 0.
        means, stds = [], []
        with torch.no_grad():
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, info = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                net_output = memb.normalizer.unnormalize(net_output_n, 1)
                ensemble_preds += net_output
                # collect the means and stds of predictions
                mean = memb.normalizer.unnormalize(info["mean_predictions"], 1)
                std = getattr(memb.normalizer, f'{1}_scaling') * info["std_predictions"] # danger
                means.append(mean)
                stds.append(std)

        ensemble_preds = ensemble_preds / float(len(self.all_models)) # delta of the state, which is the mean of the ensemble outputs
        means, stds = torch.stack(means).cpu().numpy(), torch.stack(stds).cpu().numpy()

        # proceed to the next time step
        self.cur_state = self.cur_state + ensemble_preds # the next state, TODO: use the true value for the unselected dimensions
        return_state = self.cur_state[:, self.state_idxs]
        reward = self.get_reward(return_state.cpu().numpy(), self.cur_time, shot_id=self.ref_shot_id) # next state and current time step
        self.cur_time += 1
        self.pre_action = cur_action.clone()

        # new to this env
        done = self.is_done(self.cur_time)
        done = done | (self.cur_time >= self.cur_shot_time_limit)

        if batch_size > 1:
            done = np.array([done for _ in range(batch_size)])
        else:
            reward = reward[0]

        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id=self.ref_shot_id), reward, done, {'means': means, 'stds': stds, "time_step": self.cur_time}
