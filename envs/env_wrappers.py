from copy import deepcopy
from envs.base_env import NFBaseEnv


class PlanningWrapper:
    def __init__(self, core_env: NFBaseEnv):
        self.core_env = deepcopy(core_env) # for memory safety
    
    def get_shot_length(self):
        # only run this after reset
        cur_shot_time_limit = getattr(self.core_env, 'cur_shot_time_limit', None)
        assert cur_shot_time_limit is not None

        return cur_shot_time_limit - self.core_env.cur_time
    
    def get_reference_shots(self):
        return self.core_env.get_eval_shot_list()
    
    def reset(self, shot_id):        
        return self.core_env.reset(shot_id)

    def step(self, cur_action):
        return self.core_env.step(cur_action)
    
    def seed(self, seed):
        self.core_env.seed(seed)