import torch
import numpy as np
from typing import Dict, Optional, List

from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir

def get_shots(
        data: Dict[str, np.ndarray],
        min_amt_needed: int
    ) -> List[Dict[str, np.ndarray]]:
        """Split the data into shots based on the termination signals

        Args:
            data: Data that sorted into states and actuators.
            min_amt_needed: The minimum amount of samples in a trajectory needed to include it.

        Returns:
            List of the different data, by continuous time and shot.
        """
        shots = []
        cuts = np.argwhere(data['terminals']).flatten() + 1

        # collect the first shot
        if cuts[0] > min_amt_needed:
            shots.append({k: v[0:cuts[0]] for k, v in data.items()})
        
        # collect the remaining shots
        for idx in range(len(cuts) - 1):
            left, right = cuts[idx], cuts[idx + 1]
            if right - left > min_amt_needed:
                shots.append({k: v[left:right] for k, v in data.items()})

        return shots

def get_EYX(test_shots, model_dir, ensemble_mode=False): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = test_shots[0].to(device)
    test_mask = test_shots[2].cpu().numpy().flatten()

    # load the rnn model ensemble used to synthesize data
    ensemble = load_ensemble_from_parent_dir(parent_dir=model_dir)
    all_models = ensemble.members
    for memb in all_models:
        memb.to(device)
        memb.eval()
    
    # apply the rpnn models to get mean predictions
    ensemble_preds = []
    with torch.no_grad():
        for memb in all_models:
            net_input = memb.normalizer.normalize(test_input, 0)
            net_output = memb.get_net_out((net_input, ))['mean'] # TODO: use step-by-step rollouts
            ensemble_preds.append(memb.normalizer.unnormalize(net_output, 1))
    ensemble_preds = torch.stack(ensemble_preds, dim=0)

    # post process
    if not ensemble_mode:
        EYX = ensemble_preds.mean(dim=0).reshape(-1, ensemble_preds.shape[-1]).cpu().numpy() # shape: (batch_size * seq_length, datapoint_dim) 
        EYX = EYX[test_mask > 0]
    else:
        EYX = ensemble_preds.reshape(ensemble_preds.shape[0], -1, ensemble_preds.shape[-1]).cpu().numpy() # shape: (ensemble_size, batch_size * seq_length, datapoint_dim) 
        EYX = EYX[:, test_mask > 0]

    return EYX