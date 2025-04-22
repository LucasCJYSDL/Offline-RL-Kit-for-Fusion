import pickle

#!!! what you need to specify
# which states/actuators are actually used
# obs_in_use = ["betan_EFIT01",
#                 "temp_component1", 
#                 "temp_component2", 
#                 "temp_component3", 
#                 "temp_component4", 
#                 "itemp_component1", 
#                 "itemp_component2", 
#                 "itemp_component3", 
#                 "itemp_component4", 
#                 "dens_component1", 
#                 "dens_component2", 
#                 "dens_component3", 
#                 "dens_component4", 
#                 "rotation_component1", 
#                 "rotation_component2", 
#                 "rotation_component3", 
#                 "rotation_component4", 
#                 "pres_EFIT01_component1", 
#                 "pres_EFIT01_component2", 
#                 "q_EFIT01_component1", 
#                 "q_EFIT01_component2"]

obs_in_use = ["dens_component1", 
              "dens_component2", 
              "dens_component3", 
              "dens_component4"]

# acts_in_use= ['pinj','tinj', 'bt_magnitude', 'bt_is_positive','ech_pwr_total']

acts_in_use= ['pinj','tinj', 'gasA', 'ech_pwr_total']

# functions that you do not need to modify
def state_names_to_idxs(data_path):
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_states = info['state_space']
        idxs = []
        for s in obs_in_use:
            idxs.append(all_states.index(s))
    return idxs

def actuator_names_to_idxs(data_path):
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_acts = info['actuator_space']
        idxs = []
        for a in acts_in_use:
            idxs.append(all_acts.index(a))
    return idxs

# get the indices of tracking targets in the observation space
def get_target_indices(tracking_target, obs_dim):
    indices = []
    for i in range(obs_dim):
        if obs_in_use[i].startswith(tracking_target):
            indices.append(i)
    
    return indices
