import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-expert-v2") # OpenAI gym environment name
    parser.add_argument("--envn", default="walker") # OpenAI gym environment name
    parser.add_argument("--task", default="walk") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_train_pol", default=10, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_clusters", default=100, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cuda:2", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--gamma', type=float, default=0.95) 
    parser.add_argument('--train_ratio', type=float, default=0.9) 
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--action_seq_len', type=int, default=10)
    parser.add_argument('--num_finetune_policies', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.005) 
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--M', type=int, default=16)               # support action number
    parser.add_argument('--seed_per_evaluation', type=int, default=10)
    parser.add_argument('--s', type=float, nargs="*", default=None)# guidance scale
    parser.add_argument('--method', type=str, default="CEP")
    parser.add_argument('--act', type=str, default="relu")
    parser.add_argument('--q_alpha', type=float, default=None)     
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--lr_warmup_steps', type=int, default=100)
    parser.add_argument('--target_update_interval', type=int, default=2)
    parser.add_argument("--occupancy", action="store_true",  help="predict state occupancy instead of transition function")
    parser.add_argument("--normalize", action="store_true",  help="Normalize inputs")
    parser.add_argument("--use_norm", action="store_true",  help="Use batch norm")
    parser.add_argument("--use_future_state", action="store_true",  help="Use sampled future state to train occupancy model")
    parser.add_argument("--qlearning_dataset", action="store_true",  help="Use Q learning dataset")
    parser.add_argument("--diamond", action="store_true",  help="use diamond model")
    parser.add_argument("--small", action="store_true",  help="use smaller score model")
    parser.add_argument("--velocity", action="store_true",  help="use velocity as next observation")
    parser.add_argument("--sep_action", action="store_true",  help="use separate action head")
    parser.add_argument("--use_lr_sched", action="store_true",  help="use learning rate scheduling")
    parser.add_argument("--time", action="store_true",  help="add time to state")
    parser.add_argument("--memory", action="store_true",  help="add memory to data")    
    parser.add_argument('--num_steps_denoising', type=int, default=3)
    parser.add_argument('--num_prev_states', type=int, default=1)
    parser.add_argument('--cond_channels', type=int, default=256)
    parser.add_argument('--sigma_min', type=float, default=2e-3) 
    parser.add_argument('--sigma_max', type=float, default=5.0) 
    parser.add_argument('--sigma_data', type=float, default=1.0) 
    parser.add_argument('--rho', type=int, default=7)
    parser.add_argument('--loc', type=float, default=-0.4)
    parser.add_argument('--scale', type=float, default=1.2)
    parser.add_argument('--sampling_sigma_min', type=float, default=2e-3)
    parser.add_argument('--sampling_sigma_max', type=float, default=20)
    parser.add_argument('--s_churn', type=float, default=0.0)
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--s_tmin', type=float, default=0.0)
    parser.add_argument('--s_tmax', type=float, default=float("inf"))
    parser.add_argument('--s_noise', type=float, default=1.0)
    parser.add_argument('--sigma_offset_noise', type=float, default=0.3)
    parser.add_argument("--orig", action="store_true",  help="add time to state") 
    
    parser.add_argument("--use_target", action="store_true",  help="use td learning with target network to learn the occupancy distribution.") 
    print("**************************")
    args = parser.parse_known_args()[0]
    #args.gpu_ids = [int(item) for item in args.gpu_ids.split(',')]
    '''if args.env == "fusion":
        args.expid = args.env+str(args.seed)+"_"+args.act+"_"+str(args.lr)+"_"+str(args.batch_size)+"_"+str(args.tau)
    else:
        args.expid = args.env+str(args.seed)+"_"+str(args.gamma)+"_"+str(args.target_update_interval)+"_"+args.act+"_"+str(args.lr)'''
    args.expid = args.env+str(args.seed)+"_"+args.act+"_"+str(args.lr)+"_"+str(args.batch_size)+"_"+str(args.tau)+"_s"

    #print("no weight")
    #args.expid += "_nw" 
    if not args.use_target:
        args.expid += "_mc"
    else:
        args.expid += "_td"
    if args.normalize:
        args.expid += "_nm"
    if args.use_norm:
        args.expid += "_bn"
    if args.time:
        args.expid += "_t"
    if args.orig:
        args.expid += "_orig"
    if args.small:
        args.expid = args.expid + "_small"
    if args.sep_action:
        args.expid = args.expid + "_sep_a"
    if args.velocity:
        args.expid = args.expid +"_vel"
    if args.diamond:
        args.expid = args.expid + "_diamond"
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
        args.env = "antmaze-medium-play-v2"
    if args.q_alpha is None:
        args.q_alpha = args.alpha
    print(args)
    return args
    
def add_dims(input_tensor, n):
    return input_tensor.reshape(input_tensor.shape + (1,) * (n - input_tensor.ndim))
    
def sample_sigma_training(args, n, device):
    s = torch.randn(n, device=device) * args.scale + args.loc
    return s.exp().clip(args.sampling_sigma_min, args.sampling_sigma_max)

def compute_conditioners(args, sigma):
    sigma = (sigma**2 + args.sigma_offset_noise**2).sqrt()
    c_in = 1 / (sigma**2 + args.sigma_data**2).sqrt()
    c_skip = args.sigma_data**2 / (sigma**2 + args.sigma_data**2)
    c_out = sigma * c_skip.sqrt()
    c_noise = sigma.log() / 4
    return *(add_dims(c, 2) for c in (c_in, c_out, c_skip)), add_dims(c_noise, 1)


def loss_dm_tf(args, score_model, s, a, s_, device = None):
    b = s.shape[0]
    
    sigma = sample_sigma_training(args, b, args.device) #  (B, )
    c_in, c_out, c_skip, c_noise = compute_conditioners(args, sigma) 
    offset_noise = args.sigma_offset_noise * torch.randn(b, 1, device=args.device) #(B, 1)
    noisy_s_ = s_ + offset_noise + torch.randn_like(s_) * add_dims(sigma, s_.ndim) #(B, 1)

    rescaled_s = s / args.sigma_data
    rescaled_noise = noisy_s_ * c_in
    
    model_output = score_model(rescaled_noise, c_noise, rescaled_s, a) 
    denoised = model_output * c_out + noisy_s_ * c_skip
    
    target = (s_ - c_skip * noisy_s_) / c_out
    loss = F.mse_loss(model_output, target)

    return loss


def loss_dm_occ(args, score_model, target_score_model, s, a, s_, sf, a_, d, gamma = 0.85, device = None):

    b = s.shape[0]
    sigma1 = sample_sigma_training(args, b, args.device) #  (B, )
    c_in1, c_out1, c_skip1, c_noise1 = compute_conditioners(args, sigma1) 
    
    offset_noise1 = args.sigma_offset_noise * torch.randn(b, 1, device=args.device)
    noisy_s_ = s_ + offset_noise1 + torch.randn_like(s_) * add_dims(sigma1, s_.ndim)

    rescaled_s = s / args.sigma_data
    rescaled_noise1 = noisy_s_ * c_in1
    
    model_output = score_model(rescaled_noise1, c_noise1, rescaled_s, a) 
    denoised = model_output * c_out1 + noisy_s_ * c_skip1

    target1 = (s_ - c_skip1 * noisy_s_) / c_out1
    loss1 = (1 - gamma) * F.mse_loss(model_output, target1)

    ########################################################
    rescaled_s_ = s_ / args.sigma_data
    if args.use_future_state:

        sigma2 = sample_sigma_training(args, b, args.device) #  (B, )
        c_in2, c_out2, c_skip2, c_noise2 = compute_conditioners(args, sigma2) 
        offset_noise2 = args.sigma_offset_noise * torch.randn(b, 1, device=args.device)
    
        noisy_s_future = sf + offset_noise2 + torch.randn_like(sf) * add_dims(sigma2, sf.ndim)
    
        rescaled_noise2 = noisy_s_future * c_in2
    
        model_output_f = score_model(rescaled_noise2, c_noise2, rescaled_s, a) 
        target_output_f = target_score_model(rescaled_noise2, c_noise2, rescaled_s_, a_).detach()
    
        ######TODO FROM HERE
        #denoised_f = model_output_f * c_out2 + noisy_s_future * c_skip2 
        #defnoised_target_f = target_output_f * c_out2 + noisy_s_future * c_skip2 
    
        
        loss2 = gamma * F.mse_loss(model_output_f, target_output_f)
    
    
    else:
        s_future,_ = target_model.sample_occ(s_, a_).detach()
    
        sigma2 = sample_sigma_training(args, b, args.device) #  (B, )
        c_in2, c_out2, c_skip2, c_noise2 = compute_conditioners(args, sigma2) 
    
        offset_noise2 = args.sigma_offset_noise * torch.randn(b, 1, device=args.device)
        noisy_s_future = s_future + offset_noise2 + torch.randn_like(s_future) * add_dims(sigma2, s_future.ndim)
    
        rescaled_noise2 = noisy_s_future * c_in2
        
        model_output_f = score_model(rescaled_noise2, c_noise2, rescaled_s, a) 
        denoised_f = model_output_f * c_out2 + noisy_s_future * c_skip2
        
        target2 = (s_future - c_skip2 * noisy_s_future) / c_out2
        loss2 = gamma * F.mse_loss(model_output_f, target2) 
    
    loss = loss1 + loss2

    return loss, loss1.clone().detach(), loss2.clone().detach()