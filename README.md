# Offline RL Kit

- Please refer to [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) for environment configuration. Notably, we don't require MuJoCo or D4RL.

- You would also need [dynamics-toolbox](https://github.com/LucasCJYSDL/dynamics-toolbox). This toolbox is different from Ian's, as we have changed the rpnn class.

- Please start from 'example/test_fusionenv.py' to convert raw fusion data to the format required by offline RL.

- You can run different offline RL algorithms simply by:
    ```bash
    python examples/run_XXX.py
    ```
    - You need to change several arguments in 'examples/run_XXX.py', including the raw data dir, dynamics model dir, reference shot, tracking target, etc.
    - XXX can be one of [cql, iql, edac, mcq, td3bc, combo, mobile, mopo, bambrl, rambo], where the first five algorithms are model-free and the rest are model-based.