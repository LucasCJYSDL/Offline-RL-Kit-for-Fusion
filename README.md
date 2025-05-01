# Offline RL Kit for Nuclear Fusion

## Requirements

- Please refer to [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) for environment configuration. Notably, we don't require MuJoCo or D4RL.

- You would also need [dynamics-toolbox](https://github.com/LucasCJYSDL/dynamics-toolbox). This toolbox is different from Ian's, as we have changed the rpnn class.

## Dynamics Modelling

- (Optional) As an alternative of real experiment data, we provide a script to synthesize data with an exisiting dynamics model:
    ```bash
    python dynamics/synthesize_rollouts.py
    ```
- You can train an ensemble of dynamics models by running the following command:
    ```bash
    python dynamics/train_dynamics.py
    ```
    - By default, the script will use the configuration specified by the "config_name" argument within "train_dynamics.py". You may optinally modify or add configuration files in "dynamics/cfgs".

## Policy Learning

- Please start from converting the raw fusion data to the format required by offline RL or Imitation Learning:
    ```bash
    python rl_preparation/process_raw_data.py
    ```

- You can run different offline RL algorithms simply by:
    ```bash
    python rl_scripts/run_XXX.py
    ```
    - XXX can be one of [cql, iql, edac, mcq, td3bc, combo, mobile, mopo, bambrl, rambo, rombrl], where the first five algorithms are model-free and the rest are model-based.

- You can run a goal-conditioned imitation learning algorithm by:
    ```bash
    python rl_scripts/run_gcil.py
    ```

- As an alternative, we provide scripts to run planning algorithms on the learned dynamics and distill policy functions from the planning results, which can be run by:
    ```bash
    python rl_scripts/run_YYY.py
    ```
    - YYY can be one of [mppi], to be extended.

- Please find more instructions on how to run/extend the codebase in this [tutorial](https://drive.google.com/file/d/1PVcsTshC1FaqZ9pweT0eW_SvUdu-ZaWu/view?usp=sharing).

## Evaluations

- Plots of episodic returns during the policy learning process, along with trained policy model checkpoints, will be generated and saved in the 'logs' folder.

- Dynamics training results will be generated and saved in the 'outputs' folder.

- We also provide scripts in the 'visualization' folder to visualize actuator choices made by a trained policy and its performance on plasma control tasks, such as profile tracking. Please configure the settings in 'visualization/profile_tracking.py', then run:
    ```bash
    python visualization/profile_tracking.py
    ```