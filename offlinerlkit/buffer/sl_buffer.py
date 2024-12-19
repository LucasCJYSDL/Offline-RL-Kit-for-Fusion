import numpy as np
from numpy.random import default_rng
from collections import namedtuple

SL_Transition = namedtuple('SL_Transition', ('state', 'action_list', 'action_num', 'action_dist', 'q'))

class SLReplaBuffer:
    def __init__(self, action_dim, state_dim, num_sampled_actions, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._num_sampled_actions = num_sampled_actions
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()
    
    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action_list': np.zeros((self.capacity, self._num_sampled_actions, self._action_dim), dtype='float32'),
            'action_num': np.zeros((self.capacity, ), dtype='int'),
            'action_dist': np.zeros((self.capacity, self._num_sampled_actions), dtype='float32'),
            'q': np.zeros((self.capacity, self._num_sampled_actions), dtype='float32')
        } 
    
    def push(self, transition: SL_Transition):
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        temp_dict = transition._asdict()

        for i in range(num_samples):
            i_idx = idx[i]
            i_action_num = len(temp_dict['action_list'][i])
            self._memory['action_num'][i_idx] = i_action_num
            self._memory['action_dist'][i_idx][:i_action_num] = temp_dict['action_dist'][i]
            self._memory['action_list'][i_idx][:i_action_num] = temp_dict['action_list'][i]
            self._memory['q'][i_idx][:i_action_num] = temp_dict['q'][i]
        
        self._memory['state'][idx] = temp_dict['state']
        self._memory['action_dist'][idx] = self._memory['action_dist'][idx] / self._memory['action_dist'][idx].sum(axis=1, keepdims=True)

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)
    
    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return SL_Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        # print(self._size, self.capacity) # 160456, 800000
        return self._return_from_idx(idx)