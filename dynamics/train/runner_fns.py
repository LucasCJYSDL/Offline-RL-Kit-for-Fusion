import tqdm
from models.mlp_ensemble import EnsembleDynamics
import os
import gym
import d4rl
import json
import scipy
import functools
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusion_fns.loss import loss_fn_tf, loss_fn_occ
from diffusion_fns.schedule import marginal_prob_std
from diffusion_fns.model import ScoreNet, SmallScoreNet, DiamondModel, SmallScoreNetN
from diffusion_utils import get_args
from dataset.dataset import D4RL_dataset, D4RL_occ_dataset
from dataset.utils import hard_update, soft_update, get_lr_sched
from train.runner_utils import loss_dm_tf, loss_dm_occ
from torch.optim.lr_scheduler import LambdaLR
from dataset.ope_utils import make_transitions

def run_epoch(model, loader, optimizer, train=True):
    epoch_stats = []
    for i, (obs, act, next_obs) in enumerate(loader):
        loss, stats = model.compute_dynamics_loss(obs, act, next_obs)
        if train:
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if i == 0:
            epoch_stats = {k: [v] for k, v in stats.items()}
        else:
            for k in epoch_stats.keys():
                epoch_stats[k].append(stats[k])
    
    out_stats = {k: np.stack(v).mean(0) for k, v in epoch_stats.items()}
    return out_stats

def train_model(model, lr, decay, epochs, train_loader, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    train_history = []
    test_history = []
    bar = tqdm.tqdm(range(epochs))
    for e in bar:
        train_stats = run_epoch(model, train_loader, optimizer, train=True)
        with torch.no_grad():
            test_stats = run_epoch(model, test_loader, optimizer, train=False)
        
        train_history.append(train_stats)
        test_history.append(test_stats)

        bar.set_description(
            "e: {}, train_loss: {:.4f}, test_loss: {:.4f}".format(
                e+1, train_history[-1]['loss'], test_history[-1]['loss']
            )
        )
    return model, train_history, test_history


def train_diffusion_model(args, dataset = None, data_loader = None):
    # The diffusion behavior training pipeline is copied directly from https://github.com/ChenDRAG/SfBC/blob/master/train_behavior.py
    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.env))):
        os.makedirs(os.path.join("./models_rl", str(args.env)))

    if not os.path.exists(os.path.join("./logs", str(args.env))):
        os.makedirs(os.path.join("./logs", str(args.env)))
    
    if not os.path.exists(os.path.join("./models_rl", str(args.env), str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.env), str(args.expid)))

    if not os.path.exists(os.path.join("./logs", str(args.env), str(args.expid))):
        os.makedirs(os.path.join("./logs", str(args.env), str(args.expid)))
        
    writer = SummaryWriter("./logs/" + str(args.env)+"/"+ str(args.expid))
    with open('./models_rl/'+str(args.env)+"/"+str(args.expid)+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    #env = gym.make(args.env)
    #env.seed(args.seed)
    #env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = dataset.obs.shape[-1] #env.observation_space.shape[0]
    action_dim = dataset.act.shape[-1] #env.action_space.shape[0]
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.diamond:
        print("Using Diamond Model.")
        score_model = DiamondModel(args, state_dim, state_dim).to(args.device)
        target_score_model = DiamondModel(args, state_dim, state_dim).to(args.device)
    else:
        if not args.small:
            score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
            target_score_model = ScoreNet(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)    
        else:
            if not args.sep_action:
                score_model= SmallScoreNet(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
                target_score_model = SmallScoreNet(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)    
            else:
                score_model= SmallScoreNetN(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
                target_score_model = SmallScoreNetN(input_dim=state_dim+action_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)     
                
    hard_update(target_score_model, score_model)
    assert dataset is not None, "Dataset cannot be None, should be an instance of CustomDiffusionDataset."
    assert data_loader is not None, "Dataloader cannot be none. It should be created from CustomDiffusionDataset."

    print("training behavior")
    model = run_diffusion_epochs(args, score_model, target_score_model, data_loader, start_epoch=0)
    print("finished")
    return model

def run_diffusion_epochs(args, score_model, target_score_model, data_loader, start_epoch=0):
    #def datas_():
    #    while True:
    #        yield from data_loader
    #datas = datas_()
    n_epochs = args.n_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    save_interval = 50
    
    optimizer = torch.optim.Adam(score_model.parameters(), lr=args.lr)
    lr_sched = get_lr_sched(optimizer, args.lr_warmup_steps) if args.use_lr_sched else None

    for epoch in tqdm_epoch:
        avg_loss, avg_loss1, avg_loss2 = 0., 0., 0.
        num_items = 0
        for step, data in enumerate(data_loader): #for _ in range(1000):
            #data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}

            s = data['s'] #is Bxstate_dim
            a = data['a']
            s_ = data['s_'] 
            a_ = data['a_']
            d = data['d']
            sf = data['sf']  #is BxNxstate_dim

            if not args.diamond:
                loss, loss1, loss2 = step_fn_orig(args, step, score_model, target_score_model, optimizer, s, a, s_, sf, a_, d)
            else:
                loss, loss1, loss2 = step_fn_diamond(args, step, score_model, target_score_model, optimizer, s, a, s_, sf, a_, d)
            if lr_sched is not None:
                lr_sched.step()
            avg_loss += loss
            if args.occupancy:
                avg_loss1 += loss1
                avg_loss2 += loss2
            num_items += 1
            if args.small:
                if step == 1000:
                    break
                
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if args.occupancy:
            wandb.log({"Average Loss": avg_loss/num_items, "Average Loss1": avg_loss1/num_items, "Average Loss 2": avg_loss2/num_items})
        else:
            wandb.log({"Average Loss": avg_loss/num_items})
            
        if (epoch % save_interval == (save_interval - 1)) or epoch == 599:
            torch.save(score_model.state_dict(), os.path.join("./models_rl", args.env, str(args.expid), "behavior_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

    return score_model

def add_dim_flatten(x, num_repeat):
    return torch.repeat_interleave(x.unsqueeze(dim = -2), num_repeat, dim = -2).view(-1, x.shape[-1])
    
def step_fn_orig(args, step, score_model, target_model, optimizer, s, a, s_, sf, a_, d):

    num_sf = sf.shape[1]
    losses1, losses1, losses2 = [], [], []
    s, a, s_, a_ = add_dim_flatten(s, num_sf), add_dim_flatten(a, num_sf), add_dim_flatten(s_, num_sf), \
    add_dim_flatten(a_, num_sf)
    d = torch.repeat_interleave(d.unsqueeze(dim = -1), num_sf, dim = -1).view(-1)
    sf = sf.view(-1, sf.shape[-1])
    #print(s.shape, a.shape, s_.shape, a_.shape, sf.shape, d.shape)
    #for step in range(num_sf_samples):
    if args.occupancy:
        conditions = [torch.cat([s, a], dim = -1), torch.cat([s_, a_], dim = -1)]
        loss, loss1, loss2 = loss_fn_occ(args, score_model, target_model, s_, sf, d, args.marginal_prob_std_fn, gamma = args.gamma, conditions = conditions, device = args.device) 
        if(step%args.target_update_interval == 0):
            soft_update(target_model, score_model, tau = args.tau)
    else:
        score_model.condition = torch.cat([s, a], dim = -1)
        loss = loss_fn_tf(args, score_model, s_, args.marginal_prob_std_fn, device = args.device)    
        score_model.condition = None
        loss1, loss2 = None, None
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()   
    return loss, loss1, loss2

def step_fn_diamond(args, step, score_model, target_model, optimizer, s, a, s_, sf, a_, d):
    
    num_sf_samples = sf.shape[1]
    losses1, losses1, losses2 = [], [], []
    for step in range(num_sf_samples):
            if args.occupancy:
                loss, loss1, loss2 = loss_dm_occ(args, score_model, target_model, s, a, s_, sf[:, step], a_, d, gamma = args.gamma, device = args.device)
                if(step%args.target_update_interval == 0):
                    soft_update(target_model, score_model)
            else:
                loss = loss_dm_tf(args, score_model, s, a, s_, device = args.device)    
                loss1, loss2 = None, None
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            losses.append(loss)
            losses1.append(loss1)
            losses2.append(loss2)
        
    loss = sum(losses)/len(losses)
    loss1 = sum(losses1)/len(losses1) if losses1[0] is not None else None
    loss2 = sum(losses2)/len(losses2) if losses2[0] is not None else None   
    return loss, loss1, loss2