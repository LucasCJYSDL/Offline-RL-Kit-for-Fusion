import numpy as np
import torch
import torch.nn as nn

from typing import Dict
from offlinerlkit.policy import BasePolicy


class BCPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        stochastic_actor: bool
    ) -> None:

        super().__init__()
        self.actor = actor
        self.actor_optim = actor_optim
        self._stochastic_actor = stochastic_actor
    
    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            if not self._stochastic_actor:
                action = self.actor(obs)
            else:
                dist = self.actor(obs)
                if deterministic:
                    squashed_action, raw_action = dist.mode()
                else:
                    squashed_action, raw_action = dist.rsample()
                action = squashed_action

        return action.cpu().numpy()
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions = batch["observations"], batch["actions"]
        
        if not self._stochastic_actor:
            a = self.actor(obss)
            actor_loss = ((a - actions).pow(2)).mean()
        else:
            dist = self.actor(obss)
            log_probs = dist.log_prob(actions)  # log-probabilities of the actions
            actor_loss = -log_probs.mean()  # negative log-likelihood

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "loss/actor": actor_loss.item()
        }