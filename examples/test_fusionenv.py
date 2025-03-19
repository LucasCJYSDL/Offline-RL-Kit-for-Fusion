import torch.nn as nn
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.fusion import SA_processor, NFEnv, get_raw_data, store_offline_dataset, load_offline_data

print("====== above is all warnings lol =======")

# device configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DummyPolicyNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.activation = nn.Tanh()
        # extra dimension time index appended at end

    def forward(self, x):
        x = self.activation(self.linear(x))
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
    raw_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_flat_top/"
    model_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar"
    tracking_target = "betan_EFIT01"
    reference_shot = 189268

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
    action_bound_path = current_dir + "/envs/actuator_bounds/noshape_gas.yaml"
    # data_dir = current_dir + '/data'
    # os.makedirs(data_dir, exist_ok=True)   
    data_path = raw_data_dir + 'processed_data_rl.h5'
    
    # load up the offline/raw data
    if not os.path.exists(data_path):
    # if True:
        print("data path doesn't exist, processing raw data")
        offline_dst = get_raw_data(raw_data_dir, reference_shot, action_bound_path)  # load the raw data and convert it to nf_data.h5
        store_offline_dataset(offline_dst, data_path, reference_shot, model_dir, device)
    
    offline_data = load_offline_data(data_path, raw_data_dir, tracking_target)

    sa_processor = SA_processor(bounds=(offline_data['action_lower_bounds'], offline_data['action_upper_bounds']), \
                                time_limit=offline_data['tracking_ref'].shape[0], device=device)
    
    env = NFEnv(model_dir, device, offline_data['tracking_ref'], offline_data['tracking_states'], offline_data['tracking_pre_actions'], offline_data['tracking_actions'], 
                offline_data['index_list'], sa_processor)
    env.seed(seed=0)
    policy = DummyPolicyNet(offline_data['obs_dim']+1, offline_data['act_dim']) 

    epi_return = evaluate(policy, env)
    print('episode return:', epi_return) 
    
