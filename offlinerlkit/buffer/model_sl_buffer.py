import numpy as np

class ModelSLReplayBuffer:
    def __init__(self):
        self._size = 0

        self.net_input = None
        self.net_output = None
        self.mask = None
        self.len_list = []
    
    def load_datasets(self, dataset):
        max_len = -1

        temp_input = np.concatenate([dataset['full_observations'], dataset['pre_actions'], dataset['full_actions']-dataset['pre_actions']], axis=-1)
        temp_output = dataset['full_next_observations'] - dataset['full_observations']

        tot_len = temp_input.shape[0]
        temp_terminal = True
        net_input, net_output = [], []
        for t in range(tot_len):
            if temp_terminal:
                temp_input_list, temp_output_list = [], []
                temp_terminal = False
            
            temp_input_list.append(temp_input[t])
            temp_output_list.append(temp_output[t])

            temp_terminal = dataset['terminals'][t]
            if temp_terminal:
                net_input.append(temp_input_list.copy())
                net_output.append(temp_output_list.copy())
                max_len = max(max_len, len(temp_input_list))
                self._size += 1

        self.net_input = np.zeros((self._size, max_len, temp_input.shape[-1]), dtype=np.float32)
        self.net_output = np.zeros((self._size, max_len, temp_output.shape[-1]), dtype=np.float32)
        self.mask = np.zeros((self._size, max_len, 1), dtype=np.float32)

        for t in range(self._size):
            temp_len = len(net_input[t])
            self.len_list.append(temp_len)
            
            self.net_input[t][:temp_len] = net_input[t]
            self.net_output[t][:temp_len] = net_output[t]
            self.mask[t][:temp_len] = 1.0
    
    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        return {
            "net_input": self.net_input[batch_indexes],
            "net_output": self.net_output[batch_indexes],
            "mask": self.mask[batch_indexes]
        }

