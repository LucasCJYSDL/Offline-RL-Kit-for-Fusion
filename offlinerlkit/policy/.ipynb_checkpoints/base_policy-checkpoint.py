import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Union


class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def train() -> None:
        raise NotImplementedError
    
    def eval() -> None:
        raise NotImplementedError
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        raise NotImplementedError
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        raise NotImplementedError

    def get_rewards(self, model, obs, target, act, time):

        if model is not None:
            pred_state = model.predict(torch.cat([obs, act], dim = -1))
            current_q = pred_state[:, self.args.q_idxes]
            current_rot = pred_state[:, self.args.rot_idxes]
            
            if self.args.target_type == "scalar":
                current_dr = self.get_dr(current_q, current_rot)
            else:
                current_dr = current_rot

            current_target = time < 500
            current_target[current_target > 0] = self.target_dr[0]
            current_target[current_target == 0] = self.target_dr[1]
            rewards = torch.abs(torch.from_numpy(current_target) - current_dr).mean(dim = -1, keepdim  = True)

        return pred_state, reward