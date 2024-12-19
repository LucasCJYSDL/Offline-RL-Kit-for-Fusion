import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
from offlinerlkit.policy_trainer import MBPolicyTrainer


# model-based policy trainer
class BayesMBPolicyTrainer(MBPolicyTrainer):
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0
    ) -> None:
        super().__init__(policy, eval_env, real_buffer, fake_buffer, logger, rollout_setting, epoch, 
                         step_per_epoch, batch_size, real_ratio, eval_episodes, lr_scheduler, dynamics_update_freq)
    
    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_samples = self.real_buffer.sample(self._rollout_batch_size)
                    init_obss = init_samples["observations"].cpu().numpy()
                    init_priors = init_samples["priors"].cpu().numpy()

                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, init_priors, self._rollout_length) # ipt
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch) # ipt
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            if e > self._epoch - 11:
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_{}.pth".format(e)))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def batch_evaluate(self, eval_env_vectors, eval_path, ensemble_size, use_search=True, use_ba=True):
        n_evals = self._eval_episodes
        last_10_iterates = []
        for ckpt in os.listdir(eval_path):
            self.policy.load_state_dict(torch.load(os.path.join(eval_path, ckpt)))
            self.policy.eval()

            total_reward = [0.0 for _ in range(n_evals)]
            dones = [False for _ in range(n_evals)]
            
            states = []
            for i in range(n_evals):
                states.append(eval_env_vectors[i].reset())
            states = np.array(states)

            if use_search:
                all_priors = [np.array([1.0 for i in range(ensemble_size)])/float(ensemble_size) for j in range(n_evals)]
                all_priors = np.array(all_priors).T

            while True:
                # get the action
                with torch.no_grad():
                    actions, logits = self.policy.select_action(states, return_dists=True)

                if use_search:
                    logits[0] = logits[0].cpu().numpy()
                    logits[1] = logits[1].cpu().numpy()
                    tree_roots = self.policy._searcher.set_roots(states.shape[0])
                    self.policy._searcher.prepare(tree_roots, all_priors, states, logits)
                    # print("Start searching ...")
                    self.policy._searcher.search(tree_roots, self.policy.get_search_quantity, hide_tdqm=True)
                    # print("Start sampling ...")
                    actions, _, _, _ = self.policy._searcher.sample(tree_roots, deterministic=True)

                # env step
                j = 0
                next_states = []
                idx_list = []
                real_next_states = []
                reward_list = []
                for i in range(n_evals):
                    if not dones[i]:
                        state, reward, done, _ = eval_env_vectors[i].step(actions[j])
                        reward = 0.0 if reward is None else reward
                        reward_list.append(reward)
                        next_states.append(state)
                        total_reward[i] += reward
                        dones[i] = done
                        if not done:
                            real_next_states.append(state)
                            idx_list.append(j)
                        j += 1

                if use_search:
                    if len(states) > len(real_next_states):
                        print("Number of remaining envs: {}".format(len(real_next_states)))
                    # update the priors
                    if use_ba:
                        real_samples = np.concatenate([np.array(next_states) - states, np.array(reward_list)[:, np.newaxis]], axis=1)
                        _, _, _, info = self.policy.dynamics.step(all_priors, states, actions, self.policy._elite_only, self.policy._elite_list, real_samples)
                        all_probs = info['likelihood']
                        all_prod = all_probs * all_priors
                        all_priors = all_prod / (all_prod.sum(axis=0, keepdims=True).repeat(all_probs.shape[0], axis=0) + 1e-6)

                # prepare for the next step
                if len(real_next_states) == 0:
                    break
                states = np.array(real_next_states)
                if use_search:
                    all_priors= all_priors[:, idx_list]

            return_mean = np.mean(total_reward)
            return_mean = self.eval_env.get_normalized_score(return_mean) * 100
            print("Checkpoint: {}; Mean return: {}".format(ckpt, return_mean))
            last_10_iterates.append(return_mean)
        
        print(last_10_iterates)
        print("Mean {}; Std: {}".format(np.mean(last_10_iterates), np.std(last_10_iterates)))