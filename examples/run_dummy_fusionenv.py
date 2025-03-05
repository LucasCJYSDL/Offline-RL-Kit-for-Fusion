import random
import gym
# import d4rl
import numpy as np
import torch.nn as nn
import torch
import sys, os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.fusion import SA_processor, NFEnv, get_offline_data, get_raw_data
from offlinerlkit.nets import MLP

print()
print()
print("====== above is all warnings lol =======")
print()

# device configurations
device = 'cuda:0'

# temp
states_in_obs = ['betan_EFIT01',
                "temp_component1", 
                "temp_component2", 
                "temp_component3", 
                "temp_component4", 
                "itemp_component1", 
                "itemp_component2", 
                "itemp_component3", 
                "itemp_component4", 
                "dens_component1", 
                "dens_component2", 
                "dens_component3", 
                "dens_component4", 
                "rotation_component1", 
                "rotation_component2", 
                "rotation_component3", 
                "rotation_component4", 
                "pres_EFIT01_component1", 
                "pres_EFIT01_component2", 
                "q_EFIT01_component1", 
                "q_EFIT01_component2"]

acts_in_obs = ['pinj','tinj','gasA','bt_magnitude','bt_is_positive','ech_pwr_total']

class DummyPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DummyPolicyNet, self).__init__()
        self.linear = nn.Linear(input_dim + 1, output_dim, device=device)
        # extra dimension time index appended at end

    def forward(self, x):
        x = self.linear(x)
        return x

def evaluate(policy, env):    
    T = 10

    episode_return = 0
    s_t = env.reset()
    for t in range(T):
        a_t = policy.forward(s_t)
        s_t1, reward, done, _ = env.step(a_t) 
        episode_return += reward

    return episode_return

if __name__ == "__main__":
    # TODO: load these from given beta file
    offline_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_flat_top/"
    tracking_target = "betan_EFIT01"
    reference_shot = 189268
    action_bound_path = "/zfsauton2/home/linmo/FusionControl/cfgs/control/environment/actuator_bounds/noshape_gas.yaml"
    data_path = "/zfsauton2/home/linmo/Offline-RL-Kit/data/nf_data_select_states.h5"    
    model_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar"

    if False:#os.path.exists(data_path):
        print("data path exists, reading from processed data")
        offline_data = get_offline_data(data_path, tracking_target)
    else:
        print("data path doesn't exist, processing raw data")
        offline_data = get_raw_data(offline_data_dir, tracking_target, reference_shot, action_bound_path, data_path, states_in_obs, acts_in_obs)  # load the raw data and convert it to nf_data.h5

    
    sa_processor = SA_processor(bounds=(offline_data['action_lower_bounds'], offline_data['action_upper_bounds']), \
                                time_limit=offline_data['tracking_ref'].shape[0], device=device)
    env = NFEnv(model_dir, device, offline_data['tracking_ref'], offline_data['tracking_states'], offline_data['tracking_actions'], offline_data['index_list'], sa_processor)
    env.seed(seed=0)
    policy = DummyPolicyNet(offline_data['obs_dim'], offline_data['act_dim']) 
    
    # single pass through to make sure params are initiated
    dummy_input = torch.cat([torch.Tensor(offline_data['observations'][0]).to(device), torch.Tensor([1]).to(device)])
    dummy_action = policy(dummy_input)
    # print('dummy_action shape', dummy_action.shape)

    epi_return = evaluate(policy, env)
    print('episode return:', epi_return) 
    
