"""
Extract tracking targets from given trajectories.
"""
import numpy as np


def step_function_targets(obs_seq, target_idxs, terminal_seq, change_every=50):
    """
    Generate a sequence of targets that change every `change_every` steps.
    """
    tot_len = len(obs_seq)
    target_dim = len(target_idxs)

    if terminal_seq is None: # a trick to handle the difference between general data and tracking data
        terminal_seq = np.zeros((tot_len, ), dtype=bool)
        terminal_seq[-1] = True

    targets = np.zeros((tot_len, target_dim), dtype=np.float32)
    s_id = 0
    for i in range(tot_len):
        if (i+1) % change_every == 0 or terminal_seq[i] or i == (tot_len - 1):
            tmp_target = obs_seq[i][target_idxs]
            targets[s_id:(i+1)] = tmp_target # TODO: add some randomness here
            s_id = i + 1

    return targets

def fixed_ref_shot_targets(ref_obs_seq, target_idxs, terminal_seq):
    """
    Use tracking targets in a reference shot as the targets for each shot in the offline dataset.
    """
    ref_shot_len = len(ref_obs_seq)
    ref_targets = ref_obs_seq[:, target_idxs]
    padding_target = ref_targets[-1]

    if terminal_seq is None: # a trick to handle the difference between general data and tracking data
        terminal_seq = np.zeros((ref_shot_len, ), dtype=bool)
        terminal_seq[-1] = True

    tot_len = len(terminal_seq)
    # for shots longer than the reference shot, we fill in the remaining time steps with the final state
    targets = np.array([padding_target for _ in range(tot_len)]) 
    s_id = 0
    for i in range(tot_len):
        if terminal_seq[i] or i == (tot_len - 1):
            tmp_shot_len = i - s_id + 1
            fill_len = min(tmp_shot_len, ref_shot_len)
            targets[s_id:(s_id+fill_len)] = ref_targets[:fill_len]
            s_id = i + 1
    
    return targets