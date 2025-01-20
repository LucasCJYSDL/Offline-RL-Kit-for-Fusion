import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import rbf_kernel
from torch.optim.lr_scheduler import LambdaLR
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
import torch.nn as nn
import gym
import functools
import copy
import os
import torch.nn.functional as F
from tqdm import tqdm
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
import tree
from dataset.utils import sample_future_states, convert
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

    
def find_partition(index, vals, keys, first_idxes):

    '''low, high = 0, len(vals)

    while low <= high:
        mid = low + high - low //2 

        if vals[mid] == index:
            return mid

        elif vals[mid'''
    last_key = first_idxes.keys()[0]
    for key, val in first_idxes.items():
        if index >= val:
            last_key =  key
        else:
            return last_key
            
class PrepTrainDataset(Dataset):
    def __init__(self, policies, data_dir, batch_size, train = True):
        super().__init__()
        self.policies = policies
        self.batch_size = batch_size
        self.root_dir = data_dir
        self.train = train

        

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):


        policy_dir = os.path.join(self.root_dir, str(policy_idx))
        idxes = np.array(os.listdir(os.path.join(self.root_dir, str(policy_idx)))) 
        traj_idxes = np.random.choice(idxes, size=self.batch_size, replace=False)
        trajs = []
        for traj_idx in traj_idxes:
            with open(os.path.join(policy_dir, str(traj_idx)), 'rb') as file:
                    traj = pickle.load(file)
            trajs.append(traj)
        return trajs, idx

 
class OPETrainDataset(Dataset):
    def __init__(self, policies, data_dir, batch_size, train = True):
        super().__init__()
        self.policies = policies
        self.root_dir = data_dir

        self.total_num_traj = 0
        '''
        self.first_index = {}
        for policy in self.policies:
            self.first_index[policy] = self.total_num_traj
            num_traj = len(os.listdir(os.path.join(self.root_dir, str(policy))))   
            
        
        self.vals = list(self.first_index.values())
        self.keys = list(self.first_index.keys())'''

        self.traj_filenames = []
        policy_dirs = os.listdir(self.root_dir)
        for direc in policy_dirs:
            traj_filenames = os.listdir(os.path.join(self.root_dir, direc))
            for filename in traj_filenames:
                if "data" in filename:
                    continue
                else:
                    self.total_num_traj += 1
                    self.traj_filenames.append(os.path.join(self.root_dir, direc, filename))
        
    def __len__(self): return self.total_num_traj

    def __getitem__(self, idx):
     
        with open(self.traj_filenames[idx], 'rb') as file:
                traj = pickle.load(file)
        traj['s'] = np.array(traj['s'])
        traj['a'] = np.array(traj['a'])
        traj['s_'] = np.array(traj['s_'])
        traj['r'] = np.array(traj['r'])
        traj['d'] = np.array(traj['d'])
        return traj


class PrepTestDataset(Dataset): #todo
    def __init__(self, pidx, data_dir, batch_size, train = True):
        super().__init__()
        self.pidx = pidx
        self.root_dir = data_dir
        self.policy_dir = os.path.join(self.root_dir, str(pidx))
        self.traj_idxes = os.listdir(policy_dir)
        
    def __len__(self):
        return len(self.traj_idxes)

    def __getitem__(self, idx):

        policy_dir = os.path.join(self.root_dir, str(idx))      
        traj_path = os.path.join(self.policy_dir, self.traj_idxes[idx])
        with open(traj_path, 'rb') as file:
            traj = pickle.load(file)
            return traj


def unpack_dataset(trajectories, normalize = True):
    '''trajectory = {}
    for d in list_of_dicts:
        for k, v in d.items():
            result[k].append(v)'''
   
    trajectories['s'] = np.concatenate([trajectories['s'], np.expand_dims(trajectories['r'], axis = -1)], axis = -1)
    num_samples = trajectories['s'].shape[0] * trajectories['s'].shape[1]
    obs_shape = trajectories['s'].shape[-1]
    act_shape = trajectories['a'].shape[-1]

    obs = trajectories['s'].reshape(num_samples, obs_shape)[:-2]
    act = trajectories['a'].reshape(num_samples, act_shape)[:-2]
    next_obs = trajectories['s'].reshape(num_samples, obs_shape)[1:-1]
    next_act = trajectories['a'].reshape(num_samples, act_shape)[1:-1]
    terminated = trajectories['d'].reshape(-1)[:-2]
    next_next_obs = trajectories['s'].reshape(num_samples, obs_shape)[2:]
    rwd = trajectories['r'].reshape(-1, 1)[:-2]

    '''if normalize: #precompute mean and std: todo
        obs = (obs - obs_mean)/obs_std
        next_obs = (next_obs - obs_mean)/obs_std
        next_next_obs = (next_next_obs - obs_mean)/obs_std'''
        
    
    return obs, act, rwd, next_obs, next_next_obs, next_act, terminated
    
#function to return n (s, a, s', a', sf) samples given a trajectory
def make_transitions(args, trajectories, num_samples = 10, train_prep = False, num_policies = 1):

    obs, act, rwd, next_obs, next_next_obs, next_act, terminated = unpack_dataset(trajectories, normalize = args.normalize)

    next_terminal = np.zeros(obs.shape[0])
    future_t = obs.shape[0] - 1
    for i in range(obs.shape[0] - 1, -1, -1):
      if terminated[i]:
        future_t = i
      next_terminal[i] = future_t
    next_terminal = np.array(next_terminal)
    
    if train_prep:
        idxes = np.random.randint(0, len(next_terminal), int(100/num_policies))
        future_obs = sample_future_states(args, next_terminal[idxes], idxes, args.gamma, next_obs, next_next_obs, max_horizon = 200, n_samples = num_samples, train = True, use_target = False) 
    else:
        idxes = np.random.randint(0, len(next_terminal), 100)
        future_obs = sample_future_states(args, next_terminal[idxes], idxes, args.gamma, next_obs, next_next_obs, max_horizon = 200, use_target = args.use_target, n_samples = num_samples, train = True) 


    data = {}
    data['s'] = convert(obs[idxes]) #is Bxstate_dim
    data['a'] = convert(act[idxes].numpy())
    data['s_'] = convert(next_obs[idxes])
    data['a_'] = convert(next_act[idxes].numpy())
    data['d'] = convert(terminated[idxes].numpy())
    data['r'] = convert(rwd[idxes].numpy())
    data['sf'] = convert(future_obs)

    #for k, v in data.items():
    #    print(k, v.shape)
    #print(hello)
    return data #returns num_samples* 100 tuples


def load_policy(rlu_path, policy_path_rlu):

    with tf.io.gfile.GFile(rlu_path, 'r') as f:
         policy_database = json.load(f)
    policy = tf.saved_model.load(os.path.join(policy_path_rlu, policy_metadata['policy_path']))

    return policy

#computes a set of canonical states and action prototypes per policy.
def get_can_states(args, dataset, max_trajectory_length = 1000, num_samples = 1, naive = True):


    all_canonical_states = {}
    if os.path.exists(os.path.join(dataset.root_dir, "canonical_states.pkl")):
            return
    for idx in tqdm(dataset.policies):

        policy_dir = os.path.join(dataset.root_dir, str(idx))  
        if os.path.exists(os.path.join(policy_dir, "canonical_data.pkl")):
            continue
        canonical_data = {}
        s, a  = [], []
        for traj_idx in os.listdir(policy_dir):
            if "data" in traj_idx:
                continue
            with open(os.path.join(policy_dir, str(traj_idx)), 'rb') as file:
                    traj = pickle.load(file)

            if naive:
                sample = np.random.randint(0, len(traj['s']), min(num_samples, len(traj['s']))).astype(int)
                sampled_s = np.array(traj['s'])[sample]
                sampled_a = np.array(traj['a'])[sample]
                s.append(sampled_s)
                a.append(sampled_a)
                
            else:
                continue
                
         
        s = np.concatenate(s, axis = 0)
        a = np.concatenate(a, axis = 0)
        print(s.shape, a.shape)
        canonical_data['s'] = s 
        canonical_data['a'] = a
        all_canonical_states[idx] = s.copy()

        scaler = StandardScaler()
        s_scaled = scaler.fit_transform(canonical_data['s'])
        # Perform PCA
        pca = PCA(n_components= canonical_data['a'][0].shape[-1], random_state = 42)
        s_pca = pca.fit_transform(s_scaled)
        
        # Perform K-means clustering
        #here we use 100 trajs and 100 states per traj, so maybe 100 clusters to start with
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
        kmeans.fit(np.concatenate([s_pca, canonical_data['a']], axis = -1)) # 100 x (2*action_dim)

        centroids = kmeans.cluster_centers_
        print(policy_dir)
        with open(os.path.join(policy_dir, "canonical_data.pkl"), "wb") as f:
            pickle.dump(centroids, f)
        
            
    with open(os.path.join(dataset.root_dir, "canonical_states.pkl"), "wb") as f:
        pickle.dump(all_canonical_states, f)
    
    


def get_can_actions_policy(policy, can_states): #todo

    all_can_actions = []
    for key, val in can_states.items():
        
        batched_can_states = dict(tree.map_structure(lambda x: np.expand_dims(x, axis=0) if not np.isscalar(x) else np.expand_dims(x, axis=0)[np.newaxis, :], can_states[key]))
        batched_can_states = tree.map_structure(lambda x: tf.cast(tf.convert_to_tensor(x), tf.float32), batched_can_states)
        if hasattr(policy, 'initial_state'):
            can_actions = policy(batched_can_states, ((),))[0]
        else:
            can_actions = policy(batched_can_states)

    return can_actions
            
    ''' this is v hacky but it's an attempt at a good inductive bias. i.e. during training, we take
    can_actions for each policy at each epoch and then 
    all_can_actions.append(can_actions)                                  

    return can_actions'''
