import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from contextlib import nullcontext

from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir

class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # load the well-trained rnn model ensemble for training
        ensemble = load_ensemble_from_parent_dir(parent_dir=model_path) # TODO: an ensemble of dynamics models
        self.all_models = nn.ModuleList(ensemble.members)
        self.num_ensemble = len(self.all_models)

        for memb in self.all_models:
            memb.to(device)
            memb.eval()
            # print(type(memb))

        self.device = device
        self.member_list = np.array(range(0, self.num_ensemble))
    
    def reset(self, hidden_states):
        for memb in self.all_models:
            memb.reset()

        if hidden_states is not None:
            if type(hidden_states) == np.ndarray:
                hidden_states = torch.tensor(hidden_states, device=self.device, dtype=torch.float32)
            # danger
            i = 0
            for memb in self.all_models:
                memb._hidden_state = hidden_states[i].clone()
                i += 1
        
    def forward(self, net_input, is_tensor=False, with_grad=False):
        if not is_tensor:
            net_input = torch.tensor(net_input, device=self.device)
        means, stds = [], []

        context = torch.no_grad() if not with_grad else nullcontext()

        with context:
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, info = memb.single_sample_output_from_torch(net_input_n, with_grad=with_grad) # we have updated the rpnn class in the dynamics toolbox
                mean = memb.normalizer.unnormalize(info["mean_predictions"], 1)
                std = getattr(memb.normalizer, f'{1}_scaling') * info["std_predictions"] # danger
                if not is_tensor:
                    mean = mean.cpu().numpy()
                    std = std.cpu().numpy()
                means.append(mean)
                stds.append(std)

        if not is_tensor:
            return np.array(means), np.array(stds)
        return torch.stack(means), torch.stack(stds)
    
    def get_sl_loss(self, x, y, mask):
        sl_loss = 0.

        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        mask = torch.tensor(mask, device=self.device)

        for memb in self.all_models:
            memb_x = memb.normalizer.normalize(x, 0)
            net_out = memb.get_net_out((memb_x, )) 
            # memb_y = y / getattr(memb.normalizer, f'{1}_scaling') # TODO: unnormalize the net_out
            memb_y = memb.normalizer.normalize(y, 1)
            sl_loss += memb.loss(net_out, (memb_x, memb_y, mask.clone()))[0]

        return sl_loss / float(self.num_ensemble)
    
    def get_net_out(self, x, mask):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        means, stds = [], []
        for memb in self.all_models:
            memb_x = memb.normalizer.normalize(x, 0)
            net_out = memb.get_net_out((memb_x, )) # the function is for sequence data but assumes the initial hidden state is all zero.
            mean = net_out['mean']
            std = torch.sqrt(torch.exp(net_out['logvar']))
            # unnormalize
            mean = memb.normalizer.unnormalize(mean, 1)
            std = getattr(memb.normalizer, f'{1}_scaling') * std
            # mask out
            mean = mean.reshape(-1, mean.shape[-1])[mask>0]
            std = std.reshape(-1, std.shape[-1])[mask>0]
            means.append(mean)
            stds.append(std)

        return torch.stack(means), torch.stack(stds)

    def random_member_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.member_list, size=batch_size)
        return idxs
    
    def get_memory(self):
        memory = []
        for memb in self.all_models:
            memory.append(memb._hidden_state.cpu().numpy())

        return np.moveaxis(np.array(memory), source=2, destination=0)