import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.utils.data_preprocess import get_raw_data, store_offline_dataset

#!!! what you need to specify
raw_data_dir = "/home/scratch/jiayuc2/fusion_data/noshape_ech" # the raw data
training_model_dir = "/home/scratch/jiayuc2/fusion_model/rpnn_noshape_ech" # the rpnn dynamics model for training
evaluation_model_dir = "/home/scratch/jiayuc2/fusion_model/rpnn_noshape_ech" # the rpnn dynamics model for evaluation, which can be different from the training one
action_bound_file = "noshape_ech.yaml" # actuator bounds, which you probably don't need to change
reference_shot = 189268 
shot_range = 500 # we would collect shots in the range [reference_shot-shot_range, reference_shot+shot_range]
tracking_shot_range = 5 # we would test the policy on shots in the range [reference_shot-tracking_shot_range, reference_shot+tracking_shot_range]
change_every = 50 # change the tracking target every () time steps

# the processed data will be saved in the same directory as the raw data
general_data_path = raw_data_dir + '/general_data_rl.h5'
tracking_data_path = raw_data_dir + '/tracking_data_rl.h5'


if __name__ == "__main__":
    # convert raw data to rl data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline_dst = get_raw_data(raw_data_dir, reference_shot, action_bound_file, shot_range) 
    store_offline_dataset(offline_dst, training_model_dir, general_data_path, reference_shot, tracking_shot_range, tracking_data_path, device)