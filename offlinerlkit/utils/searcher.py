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
    def __init__(self, params, dynamics, state_idxs, action_idxs, sa_processor):
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
        gamma=params.gamma)

        self._cfg = EasyDict(config)
        self._dynamics = dynamics
        self._use_ba = params.use_ba
        self._search_with_hidden_state = params.search_with_hidden_state

        self._state_idxs = state_idxs
        self._action_idxs = action_idxs
        self._sa_processor = sa_processor

        print("Key Hyperparameters for MCTS: ", self._cfg.root_dirichlet_alpha, self._cfg.alpha, self._cfg.lambd, self._cfg.num_actions, self._cfg.num_states)

        # TODO: using visit_count_temperature()
        self._collect_mcts_temperature = LinearParameter(start=1.0, end=0.1, num_steps=params.epoch*params.rollout_length)
    
    def set_roots(self, root_num):
        return search_tree.Roots(root_num, self._cfg.num_actions, self._cfg.num_states)
    
    def prepare(self, roots, init_priors, init_states, policy_logits, init_full_states, init_pre_actions, init_time_steps, hidden_states, init_idxs):
        if self._search_with_hidden_state:
            init_full_states = np.concatenate([init_full_states, hidden_states.reshape(init_full_states.shape[0], -1)], axis=-1) #?
        # this step is pretty hacky, but I do not want to change the mcts core
        init_full_states = np.concatenate([init_full_states, init_idxs[:, np.newaxis]], axis=-1)

        init_time_steps = np.squeeze(init_time_steps, axis=-1).astype(int)
        roots.prepare(init_priors.T.tolist(), init_states.tolist(), policy_logits, 
                      init_full_states.tolist(), init_pre_actions.tolist(), init_time_steps.tolist())
        self.init_time_steps = init_time_steps
    
    def search(self, roots, full_action_list, terminal_list, hidden_states, get_search_quantity, hide_tdqm=False):
        
        hidden_state_dim = np.prod(hidden_states.shape[1:])
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
                ori_priors, ori_states, ori_actions, ori_full_states, ori_pre_actions, ori_time_steps, dones = \
                    search_tree.batch_traverse(roots, alpha, beta, gamma, lambd, min_max_stats_lst, results)
                
                idx_list = []
                if all(dones):
                    remaining_num = 0
                else:
                    for i in range(batch_size):
                        if not dones[i]:
                            idx_list.append(i)
                    idx_list = np.array(idx_list)
                    # (5, 5000) (5000, 22) (5000, 6) (5000, 27) (5000, 13) (5000,)
                    cur_priors = np.asarray(ori_priors)[idx_list].T
                    cur_states = np.asarray(ori_states)[idx_list]
                    cur_actions = np.asarray(ori_actions)[idx_list]
                    cur_full_states = np.asarray(ori_full_states)[idx_list]
                    cur_pre_actions = np.asarray(ori_pre_actions)[idx_list]
                    cur_time_steps = np.asarray(ori_time_steps)[idx_list]
                    cur_full_action_list = full_action_list[idx_list]
                    cur_terminal_list = terminal_list[idx_list]

                    remaining_num = len(cur_states)

                if remaining_num > 0:
                    cur_batch_idx = cur_full_states[:, -1].astype(int)
                    cur_full_states = cur_full_states[:, :-1]
                    if self._search_with_hidden_state:
                        cur_hidden_states = np.moveaxis(cur_full_states[:, -hidden_state_dim:].reshape((remaining_num, ) + hidden_states.shape[1:]), source=0, destination=2).astype(np.float32) #?
                        cur_full_states = cur_full_states[:, :(cur_full_states.shape[1]-hidden_state_dim)] #?
                    else:
                        cur_hidden_states = None # it would be very slow to search with hidden states
                    self._dynamics.reset(cur_hidden_states) #?

                    # get the full action, danger
                    indices = (cur_time_steps-self.init_time_steps[idx_list])[:, np.newaxis, np.newaxis]
                    cur_full_actions = np.squeeze(np.take_along_axis(cur_full_action_list, indices, axis=1), axis=1)
                    step_actions = self._sa_processor.get_step_action(cur_actions)
                    cur_full_actions[:, self._action_idxs] = step_actions
                    cur_time_terminals = np.squeeze(np.take_along_axis(cur_terminal_list, indices, axis=1), axis=1)

                    # state transition
                    next_state, reward, done, info = self._dynamics.step(cur_priors, cur_full_states, cur_pre_actions, cur_full_actions, \
                                                                         cur_time_steps[:, np.newaxis], cur_time_terminals, self._state_idxs, cur_batch_idx)
                    next_state = self._sa_processor.get_rl_state(next_state, cur_batch_idx + 1)
                
                    if self._use_ba:
                        cur_probs = info['likelihood']
                        # belief update
                        cur_prods = cur_probs * cur_priors
                        next_priors = cur_prods / (cur_prods.sum(axis=0, keepdims=True).repeat(cur_prods.shape[0], axis=0) + 1e-6)
                    else:
                        next_priors = cur_priors

                    logits, next_values, reward_augments, reward_penalty = get_search_quantity(cur_full_states, cur_pre_actions, cur_full_actions, cur_hidden_states, next_state, done) 
                    # print(next_values.shape, reward_augments.shape, reward.shape, logits[0].shape, logits[1].shape, reward_penalty.shape)
                    # (5000, 1) (5000, 1) (5000, 1) (5000, 6) (5000, 6) (5000, 1)
                    reward = reward + reward_augments + reward_penalty
                    next_values = next_values.squeeze(-1)
                    reward = reward.squeeze(-1)
                    next_mus = logits[0]
                    next_stds = logits[1]

                    next_full_states = info["next_full_observations"]
                    if self._search_with_hidden_state:
                        temp_hidden_states = self._dynamics.get_memory() #?
                        next_full_states = np.concatenate([next_full_states, temp_hidden_states.reshape(next_full_states.shape[0], -1)], axis=-1) #?
                    next_full_states = np.concatenate([next_full_states, (cur_batch_idx+1)[:, np.newaxis]], axis=-1)

                    next_pre_actions = cur_full_actions
                    next_time_steps = info["next_time_steps"].reshape(-1).astype(int)

                    search_tree.batch_backpropagate(next_priors.T.tolist(), next_state.tolist(), next_full_states.tolist(), next_pre_actions.tolist(), \
                                                    next_time_steps.tolist(), self._cfg.gamma, reward.tolist(), next_values.tolist(), done.tolist(), \
                                                    next_mus.tolist(), next_stds.tolist(), min_max_stats_lst, results)
                    
                else:
                    search_tree.batch_backpropagate([[]], [[]], [[]], [[]], [], self._cfg.gamma, [], [], [], [[]], [[]], min_max_stats_lst, results)
                
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