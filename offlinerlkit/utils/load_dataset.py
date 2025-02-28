import numpy as np
import torch
import collections
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from gym import spaces
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#data_path = "/home/scratch/avenugo2/FusionControl/data/preprocessed/wshapecontrol"
data_path = "/home/scratch/avenugo2/FusionControl/data/preprocessed/noshape_ech"
req_shots_path = "/home/scratch/avenugo2/FusionControl/data/tm_shots.txt"
tm_labels_path = "/home/scratch/avenugo2/FusionControl/data/tm_labels"

def max_pooling(arr, target_size):

    target_size = arr.shape[1] / pool_size
    remainder = arr.shape[1] % pool_size
    
    pooled1 = arr[:, :-remainder].reshape(arr.shape[0], target_size, -1)
    result1 = np.max(pooled1, axis=2)

    pooled2 = arr[:, -remainder:]
    result2 = np.max(pooled2, axis = 2)
    
    # Combine the results
    return np.column_stack((result1, result2))

def pca(arr, num_components=4):

    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)
    pca = PCA(n_components=num_components)
    arr_pca = pca.fit_transform(arr_scaled)
    return arr_pca, pca.explained_variance_ratio_


def set_target(target_type, q_profile, rot_profile): #currently setting as mean

    if target_type == "scalar":
        mask1 = q_profile > 1
        mask2 = q_profile > 2

        idx1 = mask1.argmax(axis=1)
        idx2 = mask2.argmax(axis=1)

        idx1[~mask1.any(axis=1)] = -1
        idx2[~mask2.any(axis=1)] = -1

        mask1 = idx1 != -1
        rot1 = rot_profile[mask1, idx1[mask1]]

        mask2 = idx2 != -1
        rot2 = rot_profile[mask2, idx2[mask2]]

        dr = rot1 - rot2
        target1 = dr.mean(axis = 0)
        target2 = target1 + dr.std(axis = 0)

    elif target_type == "downsample":
        ds_rot_profile =  max_pooling(rot_profile, args.downsample_size)
        target1 = ds_rot_profile.mean(axis = 0)
        target2 = target1 + ds_rot_profile.std(axis = 0)

    elif target_type == "pca":
        pca_rot_profile = pca(rot_profile) #currently set to 4 components
        target1 = pca_rot_profile.mean(axis = 0)
        target2 = target1 + pca_rot_profile.std(axis = 0)
    
    return [target1, target2]
    
   
def fusion_dataset(args, use_time = False):
    raw_dataset = load_from_hdf5(f"{data_path}/full.hdf5")
    #print(raw_dataset.keys())
    #for key, val in raw_dataset.items():
    #    print(key, val.shape)
    
    dones = [] 
    for i in range(len(raw_dataset['shotnum']) - 1):
       if raw_dataset['shotnum'][i] != raw_dataset['shotnum'][i+1]:
            dones.append(True)
       else:
            dones.append(False)
    dones.append(True)
    dones = np.array(dones)

    with open(f'{data_path}/info.pkl', 'rb') as f:
        metadata = pickle.load(f)

    dataset = {}

    state_vars = {}
    for i, key in enumerate(metadata['state_space']):
        state_vars[key] = i 
    if args.profile == "default":
        dataset['observations'] = raw_dataset['states']
        dataset['next_observations'] = raw_dataset['states'] + raw_dataset['next_states']
    else:
        observations = {}
        rot_profile, q_profile, ech, eccd = [], [], [], []
        
        #SETTING TARGET
        args.target_rotation = set_target(args.target_type, rot_profile, q_profile)
        
        for key, val in state_vars.items():
            if 'rotation' in key:
                rot_profile.append(raw_dataset['states'][val])
            elif 'q_profile' in key:
                q_profile.append(raw_dataset['states'][val])
            elif 'ech' in key:
                ech.append(raw_dataset['states'][val])
            elif 'eccd' in key:
                eccd.append(raw_dataset['states'][val])
            else:
                observations[key] = raw_dataset['states'][val]

        if args.profile == 'downsample':
            observations['rotation_profile'] = max_pooling(rot_profile, args.downsample_size)
            observations['q_profile'] = max_pooling(q_profile, args.downsample_size)
            if args.ec_type == "profile":
                observations['ech_profile'] = max_pooling(ech, args.downsample_size)
                observations['eccd_profile'] = max_pooling(eccd, args.downsample_size)
            else:
                observations['ech_profile'] = ech
                observations['eccd_profile'] = eccd               
        elif args.profile == 'pca':
            print("Performing PCA on rotation profile...")
            observations['rotation_profile'], ev = pca(rot_profile)
            print("Rotation profile explained variance:", ev)

            print("Performing PCA on q profile...")
            observations['q_profile'], ev = pca(q_profile)
            print("Rotation profile explained variance:", ev)

            if args.ec_type == "profile":
                print("Performing PCA on ech profile...")
                observations['ech_profile'], ev = pca(ech)
                print("Rotation profile explained variance:", ev)
    
                print("Performing PCA on eccd profile...")
                observations['eccd_profile'], ev = pca(eccd)
                print("Rotation profile explained variance:", ev)
            else:
                observations['ech_profile'] = ech
                observations['eccd_profile'] = eccd   

        observations['dr_target'] = np.repeat(np.concatenate(args.target_rotation, axis = -1), observations['rot_profile'].shape[0], axis = 0)

        #todo: get rotation profile and q profile indices
        count = 0
        for key, val in observations.items():

            if key == 'rotation_profile':
                args.rot_idxes = np.arange(count, count + observations[key].shape[-1])
            elif key == 'q_profile':
                args.q_idxes = np.arange(count, count + observations[key].shape[-1])  
                
            count += observations[key].shape[-1]
            
        dataset['observations'] = np.concatenate(observations.values().tolist(), axis = 1)


    req_actuators = set(args.actuators)
    actuators = metadata['actuator_space']
    actuator_idxes = [actuators.index[item] for item in req_actuators]
    dataset['actions'] = raw_dataset['actuators'][:, actuator_idxes] + raw_dataset['next_actuators'][:, actuator_idxes]
    

    ##############################################################################3
    
    dataset['terminals'] = dones
    dataset['rewards'] = np.zeros_like(dones)#reward should be distance between target DR and current DR.
    dataset['shot_number'] = raw_dataset['shotnum']
    dataset['time'] = raw_dataset['time'].astype(int)
    if use_time:
        time = (dataset['time'] - min(dataset['time']))/(max(dataset['time']) - min(dataset['time']))
        dataset['s'] = np.concatenate([dataset['s'], time[:, None]], axis = -1)
    print("Fusion dataset created.")
    
    return dataset
    
def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }

class FusionEnv():
    def __init__(self, dataset, dynamics_model):
        self.dataset = dataset
        self.model = None
        self.dynamics_model = dynamics_model
        self.observation_shape = dataset['observations'].shape[-1]
        as_low = np.min(dataset['actions'], axis=0)
        as_high = np.max(dataset['actions'], axis=0)
        self.action_space = spaces.Box(low= as_low, high=as_high, dtype=np.float32)
        self.cur_time = 0
        self.reward = 0
        self.rewards = None
        # self.states_subset = np.zeros(self.observation_shape) # change to env.yaml stuff later
        # self.states_subset[0] = 1

    def reset(self, num_episodes):
        self.rewards = np.zeros(num_episodes)
        # assume start from beginning
        idx = 0
        return self.dataset['actions'][idx], self.dataset['observations'][idx]
        # batch_indexes = np.random.randint(0, len(self.dataset.observations), size=num_episodes)
        # return self.dataset.actions[batch_indexes], self.dataset.observations[batch_indexes]

    def step(self, obs, act, target):

        # if model is not None:
        #     pred_next_obs = model.predict(torch.cat([obs, act], dim = -1))\
         # pred_dr = pred_next_obs[:, 19:23]
        # reward = torch.abs(target - pred_dr).mean()
        
        # dynamics model prediction
        # based on FusionControl repo:
        # model_out, _ = self.dynamics_model.predict(
        #             np.hstack([obs, act, next_act])) 
        # based on dummy model
        model_out = self.dynamics_model.predict(np.hstack([obs, act]))

        delta_state = np.abs(model_out.detach().numpy() - obs) # find difference in subset of state
        next_state = obs + delta_state

        gamma = np.linalg.norm(np.abs(next_state - target)) # using l2 norm for now, can design in the future
        self.reward += gamma
        self.cur_time += 1
        
        return next_state, self.reward        

    #def get_normalized_score

    #def get_reward
    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks