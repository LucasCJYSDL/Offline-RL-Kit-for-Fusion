import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.utils.data_preprocess import get_raw_data, store_offlinerl_dataset

#!!! what you need to specify
raw_data_dir = "/zfsauton/project/fusion/data/organized/minimal_cakenn_v4_expand_cont0002_noq_fix" # the raw data
training_model_dir = "/zfsauton/project/fusion/models/rpnn_minimal_cakenn_nll_mse_v4_exp0002_noq_fix_final25" # the rpnn dynamics model for training
evaluation_model_dir = "/zfsauton/project/fusion/models/rpnn_minimal_cakenn_nll_mse_v4_exp0002_noq_fix_final" # the rpnn dynamics model for evaluation, which can be different from the training one
action_bound_file = "noshape_gas.yaml" # actuator bounds, which you probably don't need to change
reference_shot = 161412 # 189268 
warmup_steps = 5 # we won't involve the first () steps of each shot in the training dataset
change_every = 50 # change the tracking target every () time steps

rl_shot_list = list(range(reference_shot - 1000, reference_shot + 1000)) # these shots are used for rl training
il_shot_list = list(range(reference_shot - 100, reference_shot + 100)) # these shots are used to imitate
tracking_shot_list = list(range(reference_shot - 5, reference_shot + 5)) # we would test the policy by tracking shots in this list

# the processed data will be saved in the same directory as the raw data
rl_data_path = raw_data_dir + '/rl_data.h5'
il_data_path = raw_data_dir + '/il_data.h5'
tracking_data_path = raw_data_dir + '/tracking_data.h5'


if __name__ == "__main__":
    # convert raw data to rl data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_shots = list(set(rl_shot_list) | set(il_shot_list) | set(tracking_shot_list))
    offline_dst = get_raw_data(raw_data_dir, action_bound_file, all_shots, warmup_steps) 
    store_offlinerl_dataset(offline_dst, training_model_dir, rl_data_path, il_data_path, tracking_data_path, 
                            rl_shot_list, il_shot_list, tracking_shot_list, device)