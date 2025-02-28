import torch
import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from offlinerlkit.utils.ctree import search_tree
from offlinerlkit.utils.scheduler import LinearParameter

def select_action(visit_counts, temperature, deterministic):
    if deterministic:
        action_pos = np.argmax(visit_counts)
        return action_pos
    
    exp = 1.0 / temperature
    visit_counts = np.power(np.array(visit_counts), exp)
    action_probs = visit_counts / visit_counts.sum()
    action_pos = np.random.choice(len(visit_counts), p=action_probs)
    return action_pos

def visit_count_temperature(trained_steps, threshold_training_steps_for_final_lr_temperature):
    if trained_steps < 0.5 * threshold_training_steps_for_final_lr_temperature:
            return 1.0
    elif trained_steps < 0.75 * threshold_training_steps_for_final_lr_temperature:
        return 0.5
    else:
        return 0.25

class Searcher(object):
    def __init__(self, params, dynamics, elite_list):
        config = dict(
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=params.search_root_alpha,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
        # parameters for DPW
        alpha=params.search_alpha, # 0.5
        beta=0.5,
        # UCB parameter
        lambd=params.search_ucb_coe,
        # number of sampled actions/states at each node
        num_actions=params.search_n_actions,
        num_states=params.search_n_states, # 5
        # number of search iterations
        num_search=params.search_n_search,
        gamma=params.gamma
        )
        self._cfg = EasyDict(config)
        self._dynamics = dynamics
        self._use_ba = params.use_ba
        self._elite_only = params.elite_only
        self._elite_list = elite_list

        print("Key Hyperparameters for MCTS: ", self._cfg.root_dirichlet_alpha, self._cfg.alpha, self._cfg.lambd, self._cfg.num_actions, self._cfg.num_states)

        # TODO: using visit_count_temperature()
        self._collect_mcts_temperature = LinearParameter(start=1.0, end=0.1, num_steps=params.epoch*params.rollout_length)
    
    def set_roots(self, root_num):
        return search_tree.Roots(root_num, self._cfg.num_actions, self._cfg.num_states)
    
    def prepare(self, roots, init_priors, init_states, policy_logits):
        # noises = [np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(self._cfg.num_actions)).astype(np.float32).tolist() 
        #           for _ in range(roots.num)]
        roots.prepare(init_priors.T.tolist(), init_states.tolist(), policy_logits)
    
    def search(self, roots, get_search_quantity, hide_tdqm=False):
        with torch.no_grad():
            batch_size = roots.num

            # minimax value storage, to normalize the q value
            min_max_stats_lst = search_tree.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(self._cfg.value_delta_max)

            alpha, beta, gamma, lambd = self._cfg.alpha, self._cfg.beta, self._cfg.gamma, self._cfg.lambd

            for simulation_index in tqdm(range(self._cfg.num_search), disable=hide_tdqm):
                # prepare a result wrapper to transport results between python and c++ parts
                results = search_tree.ResultsWrapper(num=batch_size)
                """
                MCTS stage 1: traverse
                    Each simulation starts from the initial state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                ori_priors, ori_states, ori_actions, dones = search_tree.batch_traverse(roots, alpha, beta, gamma, lambd, min_max_stats_lst, results)
                # print(any(dones))
                if all(dones):
                    priors, states, actions = [], [], []
                elif any(dones):
                    priors, states, actions = [], [], []
                    for i in range(batch_size):
                        if not dones[i]:
                            priors.append(ori_priors[i])
                            states.append(ori_states[i])
                            actions.append(ori_actions[i])
                else:
                    priors = ori_priors
                    states = ori_states
                    actions = ori_actions

                if len(states) > 0:
                    # (20, 150) (150, 11) (150, 3)
                    cur_priors = np.asarray(priors).T
                    cur_states = np.asarray(states)
                    cur_actions = np.asarray(actions)
                    # state transition
                    next_state, reward, done, info = self._dynamics.step(cur_priors, cur_states, cur_actions, self._elite_only, self._elite_list)
                    if self._use_ba:
                        cur_probs = info['likelihood']
                        # belief update
                        cur_prods = cur_probs * cur_priors
                        next_priors = cur_prods / (cur_prods.sum(axis=0, keepdims=True).repeat(cur_prods.shape[0], axis=0) + 1e-6)
                    else:
                        next_priors = cur_priors

                    logits, next_values, reward_augments, reward_penalty = get_search_quantity(cur_states, cur_actions, cur_priors.T, next_state) 
                    # print(next_values.shape, reward_augments.shape, reward.shape, logits[0].shape, logits[1].shape, reward_penalty.shape)
                    # (5000, 1) (5000, 1) (5000, 1) (5000, 6) (5000, 6) (5000, 1)
                    reward = reward + reward_augments + reward_penalty
                    next_values = next_values.squeeze(-1)
                    reward = reward.squeeze(-1)
                    next_mus = logits[0]
                    next_stds = logits[1]

                    search_tree.batch_backpropagate(next_priors.T.tolist(), next_state.tolist(), self._cfg.gamma, reward.tolist(), next_values.tolist(), done.tolist(),
                                                    next_mus.tolist(), next_stds.tolist(), min_max_stats_lst, results)
                    
                else:
                    search_tree.batch_backpropagate([[]], [[]], self._cfg.gamma, [], [], [], [[]], [[]], min_max_stats_lst, results)
                
    def sample(self, roots, deterministic=False):
        roots_visit_count_distributions = roots.get_distributions()
        roots_sampled_actions = roots.get_sampled_actions()
        # print(len(roots_visit_count_distributions), len(roots_sampled_actions), roots_visit_count_distributions[0], roots_sampled_actions[0])
        # 1500 1500 [6, 7, 9, 7, 7, 7, 6, 1]
        action_list = []
        for i in range(roots.num):
            action_idx = select_action(roots_visit_count_distributions[i], temperature=self._collect_mcts_temperature.value, deterministic=deterministic)    
            action_list.append(roots_sampled_actions[i][action_idx])

        self._collect_mcts_temperature.decrease()
        return np.array(action_list), roots_visit_count_distributions, roots_sampled_actions, roots.get_target_qs()