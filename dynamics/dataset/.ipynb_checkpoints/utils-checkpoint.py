import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import rbf_kernel
from torch.optim.lr_scheduler import LambdaLR
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
# Function to compute MMD between two sets of samples
def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples X and Y.
    
    Parameters:
    X : ndarray of shape (n_samples_X, n_features)
        First set of samples.
    Y : ndarray of shape (n_samples_Y, n_features)
        Second set of samples.
    kernel : str, optional (default='rbf')
        The kernel to use. Currently, only 'rbf' is supported.
    gamma : float, optional (default=None)
        Kernel coefficient for RBF kernel. If None, defaults to 1 / (n_features * var(X)).
    
    Returns:
    mmd_value : float
        The computed MMD statistic.
    """
    n_X, n_Y = X.shape[0], Y.shape[0]
    if kernel == 'rbf':
        # Use the median heuristic for gamma if not provided
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        # Compute RBF kernel matrices
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD^2 estimate
    mmd_value = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    
    return mmd_value

def make_fusion_dataset(args, data_path):
    raw_dataset = load_from_hdf5(f"{data_path}/full.hdf5")
    print(raw_dataset.keys())

    
    dones = [] 
    for i in range(len(raw_dataset['shotnum']) - 1):
       if raw_dataset['shotnum'][i] != raw_dataset['shotnum'][i+1]:
            dones.append(True)
       else:
            dones.append(False)
    dones.append(True)
    dones = np.array(dones)
    
    dataset = {}
    dataset['observations'] = raw_dataset['states']
    dataset['actions'] = raw_dataset['actuators'] + raw_dataset['next_actuators']
    dataset['terminals'] = dones
    dataset['rewards'] = None
    dataset['sidx'] = raw_dataset['shotnum']
    dataset['time'] = raw_dataset['time'].astype(int)

    dataset['time'] = (dataset['time'] - min(dataset['time']))/(max(dataset['time']) - min(dataset['time']))
    if args.time:
        dataset['observations'] = np.concatenate([dataset['observations'], dataset['time'][:, None]], axis = -1)
    print("Fusion dataset created.")
    for key, val in dataset.items():
        if val is not None:
            print(key, val.shape)
        else:
            print(key, None)
    return dataset
    
'''
def sample_future_states(next_terminal, obs_indices, gamma, next_obs, max_horizon = None):
      horizon = next_terminal - (obs_indices + 1)
      ret_arr = np.tile(np.arange(max_horizon), (horizon.shape[0], 1))
      probs = gamma**ret_arr
      probs = (1-gamma)*probs
      mask = ret_arr <= horizon[:, None]
      probs *= mask
      probs /= np.sum(probs, axis=1, keepdims=1)
      #delta = np.apply_along_axis(np.random.choice, axis = 1, arr = ret_arr, size = 1, p=probs)    #  np.apply_along_axis(np.random.choice, axis=1, arr=x, size=2))
      delta = []
      for i in range(len(horizon)):
          if(np.isnan(probs[i]).any()):
              #print(next_terminal[i], obs_indices[i])
              delta.append(i)
          else:
              delta.append(int((np.random.choice(ret_arr[i], 1, p=probs[i])+i)))
      #delta = np.array([np.random.choice(ret_arr[i], 1, p=probs[i]) for i in range(len(horizon))])
      print(delta[:100])
      future_obs = next_obs[np.array(delta)]
      return future_obs

#given a state s_t, sample the future state distribution with probabilities starting from s_t+1
def sample_future_states(args, next_terminal, obs_indices, gamma, next_obs, max_horizon = None, n_samples = 1, avoid_terminal = False):
      
      horizon = next_terminal - (obs_indices + 1)
      ret_arr = np.tile(np.arange(max_horizon), (horizon.shape[0], 1))
      probs = gamma**ret_arr
      probs = (1-gamma)*probs
      mask = ret_arr <= horizon[:, None]
      probs *= mask
      probs /= np.sum(probs, axis=1, keepdims=1)
      #delta = np.apply_along_axis(np.random.choice, axis = 1, arr = ret_arr, size = 1, p=probs)    #  np.apply_along_axis(np.random.choice, axis=1, arr=x, size=2))
      delta = []
      for i in range(len(horizon)):
          if(np.isnan(probs[i]).any()):
              print(next_terminal[i], obs_indices[i])
              if avoid_terminal:
                  continue
              arr = np.ones(n_samples)*i
              delta.append(arr)
          else:
              delta.append((np.random.choice(ret_arr[i], n_samples, p=probs[i])+i).astype(int))
      #delta = np.array([np.random.choice(ret_arr[i], 1, p=probs[i]) for i in range(len(horizon))])
      delta = np.array(delta).flatten().astype(int)
      print(delta.shape)
      future_obs = next_obs[delta] 
      return future_obs

'''

    
#given a state s_t, sample the future state distribution with probabilities starting from s_t+1
def sample_future_states(args, next_terminal, obs_indices, gamma, next_obs, next_next_obs, max_horizon = None, n_samples = 1, avoid_terminal = False, use_target = None, train = False):

      if use_target:
          horizon = next_terminal - (obs_indices + 2)
      else:
          horizon = next_terminal - (obs_indices + 1)
      ret_arr = np.tile(np.arange(max_horizon), (horizon.shape[0], 1))
      probs = gamma**ret_arr
      probs = (1-gamma)*probs
      mask = ret_arr <= horizon[:, None]
      probs *= mask
      probs /= np.sum(probs, axis=1, keepdims=1)
      #delta = np.apply_along_axis(np.random.choice, axis = 1, arr = ret_arr, size = 1, p=probs)    #  np.apply_along_axis(np.random.choice, axis=1, arr=x, size=2))
      #for horizon = 0, the code handles the probabilities
      #for horizon = -1, we need a special case
      #for horizon = -2, we need a special case
      delta = []
      if not use_target:
          for i in range(len(horizon)):
              if(horizon[i] <= -1):
                  #print(next_terminal[i], obs_indices[i])
                  if avoid_terminal:
                      continue
                  arr = np.ones(n_samples)*i
                  delta.append(arr)
              else:
                  delta.append((np.random.choice(ret_arr[i], n_samples, p=probs[i])+i).astype(int))
          if train:
                future_obs = [next_obs[item.astype(int)] for item in delta]
                future_obs = np.array(future_obs)
          else:
              delta = np.array(delta).flatten().astype(int)
              future_obs = next_obs[delta]    
      else:
          for i in range(len(horizon)):
              if(horizon[i] <= -1): # for horizon = -1 or horizon = -2
                  #print(next_terminal[i], obs_indices[i])
                  if avoid_terminal:
                      continue
                  arr = np.ones(n_samples)*i #this is masked out in the loss, make sure it's masked out.
                  delta.append(arr)
              else: #horizon = 0 case is handled here
                  delta.append((np.random.choice(ret_arr[i], n_samples, p=probs[i])+i).astype(int))
          if train:
                future_obs = [next_next_obs[item.astype(int)] for item in delta]
                future_obs = np.array(future_obs)
          else:
              delta = np.array(delta).flatten().astype(int)
              future_obs = next_next_obs[delta]      
      #print(future_obs.shape)
      return future_obs

      '''
      else:
          for i in range(len(horizon)):
            if(next_terminal[i] == obs_indices[i]):
              arr = np.ones(n_samples)*(i+1)
              delta.append(arr)
            else:
              delta.append((np.random.choice(ret_arr[i], n_samples, p=probs[i])+i).astype(int))
          #delta = np.array([np.random.choice(ret_arr[i], 1, p=probs[i]) for i in range(len(horizon))])
          delta = np.array(delta).flatten().astype(int)
          print(delta.shape)
          future_obs = obs[delta] '''


def permutation_test(X, Y, num_permutations=1000):
    observed_mmd = compute_mmd(X, Y)
    combined = np.vstack((X, Y))
    count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        X_perm = combined[:X.shape[0]]
        Y_perm = combined[X.shape[0]:]
        permuted_mmd = compute_mmd(X_perm, Y_perm)
        if permuted_mmd >= observed_mmd:
            count += 1

    p_value = count / num_permutations
    return observed_mmd, p_value

def get_lr_sched(opt, num_warmup_steps):
    def lr_lambda(current_step):
        return 1 if current_step >= num_warmup_steps else current_step / max(1, num_warmup_steps)

    return LambdaLR(opt, lr_lambda, last_epoch=-1)
    
def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
                
def parse_stacked_trajectories(obs, act, rwd, next_obs, future_obs, terminated, next_terminal, timeout, next_act, max_eps=None):
    """ Extract episodes from stack trajectories """
    #if terminated.get_device() > -1:
    #        terminated_np = terminated.clone().cpu().numpy()
    #        eps_id = np.cumsum(terminated_np + timeout) if timeout is not None else np.cumsum(terminated_np)
    #else:
    eps_id = np.cumsum(terminated + timeout) if timeout is not None else np.cumsum(terminated)        
    eps_id = np.insert(eps_id, 0, 0)[:-1] # offset by 1 step
    max_eps = eps_id.max() + 1 if max_eps is None else max_eps
    dataset = []  
    for e in np.unique(eps_id):
        dataset.append({
            "obs": obs[eps_id == e],
            "act": act[eps_id == e],
            "rwd": rwd[eps_id == e],
            "next_obs": next_obs[eps_id == e],
            "future_obs": future_obs[eps_id == e],
            "next_act": next_act[eps_id == e] if next_act is not None else None, 
            "done": terminated[eps_id == e],
            "next_done": next_terminal[eps_id == e]
        })
        if (e + 1) >= max_eps:
            break
    return dataset

def collate_fn(batch, pad_value):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {}
    pad_batch = {k: pad_sequence([b[k] for b in batch], padding_value=pad_value) if batch[0][k] is not None else None for k in keys}
    mask = pad_sequence([torch.ones(len(b[keys[0]])) for b in batch])
    return pad_batch, mask

def unpack_mlp_dataset(args, dataset, env_name, num_prev_states = 1, max_ep_len = None, use_velocity = False):

    # unpack dataset
    print("Velocity:", use_velocity)
    #if env_name == 'antmaze-umaze' or env_name =='inverted-pendulum':
    dataset["observations"] = np.concatenate([dataset["observations"], dataset["rewards"].reshape(-1, 1)], axis = -1)
    obs = dataset["observations"][:-2]
    act = dataset["actions"][:-2]
    if use_velocity:
        next_obs = dataset["observations"][1:-1] - dataset["observations"][:-2]
    else:
        next_obs = dataset["observations"][1:-1]
    next_act = dataset["actions"][1:-1]
    rwd = dataset["rewards"].reshape(-1, 1)[:-2] if dataset['rewards'] is not None else np.zeros((dataset["observations"].shape[0] - 2, 1))
    terminated = dataset["terminals"][:-2].reshape(-1)

    next_next_obs = dataset["observations"][2:]
    if "timeouts" in dataset.keys():
        timeout = dataset["timeouts"][:-2]  
    else:
        timeout = [False] * len(terminated)
        for i, item in enumerate(timeout):
            if(i > 0 and (i+1)%max_ep_len == 0):
                timeout[i] = bool(~item)
        timeout = np.array(timeout)
        
    #else:
    #    obs = dataset["observations"]
    #    act = dataset["actions"]
    #    next_obs = dataset["next_observations"]
    #    rwd = dataset["rewards"].reshape(-1, 1)
    #    terminated = dataset["terminals"]
    #    timeout = dataset["timeouts"]
    
    print("data size:", obs.shape, act.shape, rwd.shape, next_obs.shape, next_act.shape, terminated.shape)
    
    print("num terminated:", sum(terminated))
    if timeout is not None:
        print("num timeout:", sum(timeout))
        print("num terminated and timeout:", sum(terminated & timeout))

    
    return obs, act, rwd, next_obs, next_next_obs, next_act, terminated, timeout

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    chunks = []
    for i in range(0, len(l), n): 
        chunks.append(l[i:i + n][None, :])
    return np.concatenate(chunks[:-1], axis = 0)


def get_timeseries_data(obs, act, rwd, next_obs, next_act, terminated, timeout, seq_length = 10):
        obs = divide_chunks(obs, seq_length)
        act = divide_chunks(act, seq_length)
        rwd = divide_chunks(rwd, seq_length)
        next_obs = divide_chunks(next_obs, seq_length)
        next_act = divide_chunks(next_act, seq_length)
        terminated = divide_chunks(terminated, seq_length)
        timeout = divide_chunks(timeout, seq_length)
        return obs, act, rwd, next_obs, next_act, terminated, timeout
    
def convert(val):
    return torch.from_numpy(val).to(torch.float32)

def normalize(obs, act, next_obs, future_obs, next_act, rwd, done):

    obs_train, act_train, next_obs_train = convert(obs), convert(act), convert(next_obs)
    future_obs_train, next_act_train, rwd_train = convert(future_obs), convert(next_act), convert(rwd)
    done = convert(done)

    tuples = (0)
    obs_mean = obs_train.mean(tuples)
    obs_std = obs_train.std(tuples)
    
    rwd_mean = rwd_train.mean(tuples)
    rwd_std = rwd_train.std(tuples)

    act_mean = act_train.mean(tuples)
    act_std = act_train.std(tuples)
    
    obs_train_norm = (obs_train - obs_mean) / obs_std if normalize else obs_train    
    next_obs_train_norm = (next_obs_train - obs_mean) / obs_std if normalize else next_obs_train
    future_obs_train_norm = (future_obs_train - obs_mean) / obs_std if normalize else future_obs_train   

    return obs_train_norm, act_train, next_obs_train_norm, future_obs_train_norm, next_act_train, rwd_train, done
    
def create_dataset_split(num_samples, train_ratio, obs, act, next_obs, future_obs, next_act, rwd, done, next_done, normalize = None, memory = False):
    num_train = int(num_samples * train_ratio)
    idx = np.arange(len(obs))
    np.random.shuffle(idx)
    print("idx length", len(idx))
    
    idx_train = idx[:num_train]
    idx_test = idx[num_train:num_samples]

    obs_train = convert(obs[idx_train])
    act_train = convert(act[idx_train])
    next_act_train = convert(next_act[idx_train])
    next_obs_train = convert(next_obs[idx_train])
    future_obs_train = convert(future_obs[idx_train])
    rwd_train = convert(rwd[idx_train])
    done_train = convert(done[idx_train])
    next_done_train = convert(next_done[idx_train])
    
    obs_test = convert(obs[idx_test])
    act_test = convert(act[idx_test])
    next_act_test = convert(next_act[idx_test])
    next_obs_test = convert(next_obs[idx_test])
    future_obs_test = convert(future_obs[idx_test])
    rwd_test = convert(rwd[idx_test])
    done_test = convert(done[idx_test])
    next_done_test = convert(next_done[idx_test])
    
    # normalize
    if memory:
        tuples = (0, 1)
    else:
        tuples = (0)

    obs_mean = obs_train.mean(tuples)
    obs_std = obs_train.std(tuples)
    
    rwd_mean = rwd_train.mean(tuples)
    rwd_std = rwd_train.std(tuples)

    act_mean = act_train.mean(tuples)
    act_std = act_train.std(tuples)


    print(obs_mean.shape)
    
    obs_train_norm = (obs_train - obs_mean) / obs_std if normalize else obs_train
    act_train_norm = act_train#(act_train - act_mean) / act_std if normalize else act_train    
    next_obs_train_norm = (next_obs_train - obs_mean) / obs_std if normalize else next_obs_train
    future_obs_train_norm = (future_obs_train - obs_mean) / obs_std if normalize else future_obs_train    
    
    obs_test_norm = (obs_test - obs_mean) / obs_std if normalize else obs_test
    act_test_norm = act_test #(act_test - act_mean) / act_std if normalize else act_test
    next_obs_test_norm = (next_obs_test - obs_mean) / obs_std if normalize else next_obs_test
    future_obs_test_norm = (future_obs_test - obs_mean) / obs_std if normalize else future_obs_test
    
    rwd_train_norm = (rwd_train - rwd_mean) / rwd_std if normalize else rwd_train
    rwd_test_norm = (rwd_test - rwd_mean) / rwd_std if normalize else rwd_test
    
    print("data size:", obs.shape, act.shape, next_obs.shape)
    print("train data size:", obs_train.shape, act_train.shape, next_obs_train.shape, future_obs_train.shape, next_act_train.shape, done_train.shape)
    print("train data types:", obs_train.dtype, act_train.dtype, next_obs_train.dtype, future_obs_train.dtype, next_act_train.dtype, done_train.dtype)
    print("test data size:", obs_test.shape, act_test.shape, next_obs_test.shape, future_obs_test.shape, next_act_test.shape, done_test.shape)
    
    print("obs mean", obs_mean)
    print("obs std", obs_std)
    
    print("obs_train_norm mean", obs_train_norm.mean(tuples))
    print("obs_train_norm std", obs_train_norm.std(tuples))


    print("act mean", act_mean)
    print("act std", act_std)
    
    print("act_train_norm mean", act_train_norm.mean(tuples))
    print("act_train_norm std", act_train_norm.std(tuples))

    print("act_test_norm mean", act_test_norm.mean(tuples))
    print("act_test_norm std", act_test_norm.std(tuples))
    
    print("rwd mean", rwd_mean)
    print("rwd std", rwd_std)
    
    print("rwd_train_norm mean", rwd_train_norm.mean(tuples))
    print("rwd_train_norm std", rwd_train_norm.std(tuples))
    
    print("rwd_test_norm mean", rwd_test_norm.mean(tuples))
    print("rwd_test_norm std", rwd_test_norm.std(tuples))

    train_data =  (obs_train_norm, act_train_norm, next_obs_train_norm, future_obs_train_norm, rwd_train_norm, next_act_train, done_train, next_done_train, idx_train, obs_mean, obs_std, rwd_mean, rwd_std)
    test_data = (obs_test_norm, act_test_norm, next_obs_test_norm, future_obs_test_norm, rwd_test_norm, next_act_test, done_test, next_done_test, idx_test)
    return train_data, test_data