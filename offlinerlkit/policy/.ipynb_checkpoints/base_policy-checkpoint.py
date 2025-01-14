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

    def get_rewards(self, model, obs, target, act):

        if model is not None:
            pred_state = model.predict(torch.cat([obs, act], dim = -1))

        pred_dr = pred_state[:, 19:23]
        reward = torch.abs(target - pred_dr).mean(dim = -1, keepdim  = True)
        return pred_state, reward