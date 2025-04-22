import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import torch
import h5py
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.utils.data_preprocess import get_raw_data
from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir

def synthesize_rollouts(offline_dst, raw_data_dir, model_dir, data_dir, device):
    dataset = {'observations': [], 'pre_actions': [], 'action_deltas': [], 'observation_deltas': [], 'terminals': [], 'shotnum': []}
    
    # load the rnn model ensemble for data synthesize
    ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir)
    all_models = ensemble.members
    for memb in all_models:
        memb.to(device)
        memb.eval()
    
    # generate new rollouts using raw actions, time-costly!
    shot_num_list = list(offline_dst['ref_start_index'].keys())
    for cur_shot in tqdm(shot_num_list):
        t = offline_dst['ref_start_index'][cur_shot][0]

        cur_state = torch.FloatTensor(offline_dst['observations'][t]).to(device)
        pre_action = torch.FloatTensor(offline_dst['pre_actions'][t]).to(device)
        action_delta = torch.FloatTensor(offline_dst['action_deltas'][t]).to(device)
        
        while True:                
            # get the ensemble output
            net_input = torch.cat([cur_state, pre_action, action_delta], dim=-1).unsqueeze(0)
            ensemble_preds = 0.
            with torch.no_grad():
                for memb in all_models:
                    net_input_n = memb.normalizer.normalize(net_input, 0)
                    net_output_n, _ = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                    net_output = memb.normalizer.unnormalize(net_output_n, 1)
                    ensemble_preds += net_output
            state_delta = ensemble_preds[0] / float(len(all_models)) # delta of the state, which is the mean of the ensemble outputs

            # record the output
            dataset['observations'].append(cur_state.cpu().numpy()) # TODO: only collect part of the state/action dimensions
            dataset['pre_actions'].append(pre_action.cpu().numpy())
            dataset['action_deltas'].append(action_delta.cpu().numpy())
            dataset['observation_deltas'].append(state_delta.cpu().numpy())
            dataset['terminals'].append(offline_dst['terminals'][t])
            dataset['shotnum'].append(cur_shot)

            t += 1
            if offline_dst['shotnum'][t] != cur_shot:
                break

            if offline_dst['terminals'][t-1]: # WEIRD, terminals may be true even when the shoitnum does not change
                # end of shot
                cur_state = torch.FloatTensor(offline_dst['observations'][t]).to(device)
            else:
                # otherwise, prepare for the next time step
                cur_state = cur_state + state_delta
            
            pre_action = torch.FloatTensor(offline_dst['pre_actions'][t]).to(device)
            action_delta = torch.FloatTensor(offline_dst['action_deltas'][t]).to(device)
    
    # post process
    for k in dataset:
        dataset[k] = np.array(dataset[k])
        print(k, dataset[k].shape) # for sanity check
    
    # applying the actuator bounds, optional
    dataset['action_lower_bounds'] = offline_dst['action_lower_bounds'].copy()
    dataset['action_upper_bounds'] = offline_dst['action_upper_bounds'].copy()

    # If you want to apply the bounds here, you should also apply them to the input of the dynamics models.
    # dataset['pre_actions'] = np.clip(dataset['pre_actions'], dataset['action_lower_bounds'], dataset['action_upper_bounds'])
    
    # save the synthesized data
    os.makedirs(data_dir, exist_ok=True)
    with h5py.File(data_dir + '/full.hdf5', 'w') as hdf:
        for key, value in dataset.items():
            hdf.create_dataset(key, data=value)
    
    # save the info.pkl
    with open(raw_data_dir + '/info.pkl', 'rb') as file:
        data_info = pickle.load(file) # you need to edit the info here if the state/actuator space has been changed
    
    with open(data_dir + '/info.pkl', 'wb') as f:
        pickle.dump(data_info, f)
    

if __name__ == "__main__":
    #!!! what you need to specify 
    raw_data_dir = "/home/scratch/jiayuc2/fusion_data/noshape_ech" # the raw data
    model_dir = "/home/scratch/jiayuc2/fusion_model/rpnn_noshape_ech" # the rpnn dynamics model to synthesize data
    action_bound_file = "noshape_ech.yaml" # actuator bounds, which you probably don't need to change
    reference_shot = 189268 
    shot_list = list(range(reference_shot - 50, reference_shot + 50)) # list of shots for training dynamics
    warmup_steps = 5 # we won't involve the first () steps of each shot in the training dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # directory to store the synthesized data
    synthesized_data_dir = raw_data_dir + '_synthesized'

    # preprocess the raw data 
    offline_dst = get_raw_data(raw_data_dir, action_bound_file, shot_list, warmup_steps) 
    # synthesize rollouts based on the learned dynamics models
    synthesize_rollouts(offline_dst, raw_data_dir, model_dir, synthesized_data_dir, device)