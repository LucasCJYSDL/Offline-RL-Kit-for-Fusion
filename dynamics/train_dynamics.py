import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HYDRA_FULL_ERROR"] = "1"

import warnings
warnings.filterwarnings("ignore")

import ray, time
import hydra, pickle
import numpy as np
from dynamics.utils import get_EYX
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.seed import seed_everything
from dynamics_toolbox.utils.lightning.constructors import construct_all_pl_components_for_training


@ray.remote(num_gpus=1/6)  # Request 1/6 of a GPU
def train_single_model(cfg: DictConfig, ensemble_id: int, exp_id: int):
    save_dir = os.path.join(os.getcwd(), "exp_{}".format(exp_id))
    # Clone the config for this ensemble member
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        # Update seed & save directory for uniqueness
        cfg["seed"] = cfg["seed"] + cfg.get("ensemble_size") * exp_id + ensemble_id
        cfg["data_module"]["seed"] = cfg["seed"]
        cfg["data_module"]["save_dir"] = os.path.join(save_dir, str(cfg["seed"]))
        cfg["save_dir"] = os.path.join(save_dir, str(cfg["seed"]), "model")

    # you should either comment the following line or assign random seeds (i.e., cfg["seed"]) based on the exp_id
    # to ensure that the training result of each exp is different
    seed_everything(cfg["seed"])

    # get all components
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg) # logger.savedir is the same as cfg["save_dir"]
    # print(OmegaConf.to_yaml(cfg))
    save_path = os.path.join(save_dir, str(cfg["seed"]))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_path, "config.yaml"))

    # start the training process
    trainer.fit(model, data)

    # return the test_shots and ensemble_dir, which are shared by all ensemble members
    test_shots, ensemble_dir = None, None
    if ensemble_id == 0:
        test_shots = data._te_dataset.tensors # a little buggy
        ensemble_dir = save_dir

    # evluations
    if data.test_dataloader() is not None:
        test_dict = trainer.test(model, datamodule=data, ckpt_path="best")[0]
        with open(os.path.join(save_path, "test_results.txt"), "w") as f:
            f.write("\n".join([f"{k}: {v}" for k, v in test_dict.items()]))
        tune_metric = cfg.get("tune_metric", "test/loss")
        return_val = test_dict[tune_metric]
        if cfg.get("tune_objective", "minimize") == "maximize":
            return_val *= -1
        return return_val, test_shots, ensemble_dir
    else:
        return 0, test_shots, ensemble_dir


@hydra.main(
    config_path=f'{os.path.dirname(__file__)}/cfgs',
    config_name="rpnn_noshape_ech" #!!! what you need to specify
) # load arguments from the config file

def train_ensemble(cfg: DictConfig) -> None:
    """Train an ensemble of dynamics models."""

    # prepare
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])
    with open_dict(cfg):
        # set the name of each dim of the output
        with open(os.path.join(cfg["data_path"], 'info.pkl'), 'rb') as f:
            info = pickle.load(f)
            cfg['model']['dim_name_map'] = info["state_space"]
        # set other hyperparameters
        # if "gpus" in cfg:
        #     cfg["gpus"] = str(cfg["gpus"])

    num_models = cfg.get("ensemble_size")
    outer_loop_size = cfg.get("outer_loop_size")
    ray_env_var = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    start_time = time.time()
    within_bounds = 0.0
    for outer_iter in range(outer_loop_size):
        # sequential version
        # results = [] 
        # for i in range(num_models):
        #     print(f"\n=== Training model {i + 1}/{num_models} ===")
        #     result = train_single_model(cfg, ensemble_id=i)
        #     results.append(result)
        # parallel version
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": ray_env_var}}, log_to_driver=False)
        results = ray.get([train_single_model.remote(cfg, i, outer_iter) for i in range(num_models)])
        ray.shutdown() # TODO: remove this and put ray.init outside to improve speed?

        # collect/analyze training results
        metrics = []
        for i in range(num_models):
            metrics.append(results[i][0])
            if results[i][1] is not None: 
                if outer_iter == 0: # since the test shots and gt_model are shared among all repeated experiments
                    test_shots = results[i][1]
                    gt_EYX = get_EYX(results[i][1], cfg["gt_model_path"])
                learned_ensemble_dir = results[i][2]
        
        # get ensemble predictions on the test dataset
        ensemble_EYX = get_EYX(test_shots, learned_ensemble_dir, ensemble_mode=True)
        # check if gt_EYX falls within the 95% quantile interval
        lower_bound = np.percentile(ensemble_EYX, 2.5, axis=0) 
        upper_bound = np.percentile(ensemble_EYX, 97.5, axis=0)  
        within_bounds += ((gt_EYX >= lower_bound) & (gt_EYX <= upper_bound)).astype(float)  # Shape: (batch_size, datapoint_dim)

        print("=== Ensemble Training at Iteration {} Complete ===".format(outer_iter))
        print("Test loss for each ensemble member:", metrics)
        # get eta
        elapsed_time = time.time() - start_time
        remaining_iters = outer_loop_size - (outer_iter + 1)
        eta = elapsed_time / (outer_iter + 1) * remaining_iters
        print("ETA: {} hours".format(eta/3600.0))
        # check within_bounds
        average_within_bounds = within_bounds / float(outer_iter+1)
        print("Average within_bounds ratio:", average_within_bounds.mean())
        print("Average within_bounds ratio at each dimension:", average_within_bounds.mean(axis=0))
        print("\n")


if __name__ == "__main__":
    train_ensemble()