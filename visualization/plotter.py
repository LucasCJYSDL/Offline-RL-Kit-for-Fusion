import os
import matplotlib.pyplot as plt

# TODO: the first two functions can be merged

def plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot_id, log_folder):
    n = len(quan_names)  
    rows = (n + 1) // 2  # calculate the number of rows needed for 2 subfigures per row
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), sharex=True, sharey=False)
    axes = axes.flatten()  # flatten the 2D array of axes for easier indexing

    for i in range(n):
        axes[i].plot(time_array, [rq[i] for rq in real_quan_array], label="Real")
        axes[i].plot(time_array, [cq[i] for cq in cur_quan_array], label="Agent")
        axes[i].plot(time_array, [tq[i] for tq in target_quan_array], label="Target", linestyle="--")
        axes[i].set_title(f"{quan_names[i]}", fontsize=15)
        axes[i].set_xlabel("Time", fontsize=15)
        axes[i].set_ylabel("Value", fontsize=15)
        axes[i].legend(fontsize=15)
        axes[i].grid()
        axes[i].tick_params(axis='both', which='major', labelsize=15)  # enlarged tick labels

    # remove unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # save the figure 
    save_path = os.path.join(log_folder, f"{shot_id}_tracking_quantities.png")
    plt.savefig(save_path)
    plt.close(fig)  

def plot_actions(time_array, real_act_array, cur_act_array, act_names, shot_id, log_folder):
    m = len(act_names)  
    rows = (m + 1) // 2  
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), sharex=True, sharey=False)
    axes = axes.flatten() 

    for i in range(m):
        axes[i].plot(time_array, [ra[i] for ra in real_act_array], label="Real")
        axes[i].plot(time_array, [ca[i] for ca in cur_act_array], label="Agent")
        axes[i].set_title(f"{act_names[i]}", fontsize=15)  
        axes[i].set_xlabel("Time", fontsize=15)  
        axes[i].set_ylabel("Value", fontsize=15) 
        axes[i].legend(fontsize=15)  
        axes[i].grid()
        axes[i].tick_params(axis='both', which='major', labelsize=15) 

    # remove unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(log_folder, f"{shot_id}_actions.png")
    plt.savefig(save_path)
    plt.close(fig)