import torch
def loss_fn_tf(args, model, x, marginal_prob_std, eps=1e-3, condition = None, device = None):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
        time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    alpha_t, std = marginal_prob_std(random_t, device = device)
    perturbed_x = x * alpha_t[:, None] + z * std[:, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1,)))
    return loss


def loss_fn_occ(args, model, target_model, x, xf, done, marginal_prob_std, eps=1e-3, gamma = 0.85, conditions = None, device = None):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
        time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    if args.use_target:
            #sa, sa_ = conditions[0], conditions[1]
            model.condition = conditions[0]
            random_t1 = torch.rand(x.shape[0], device=device) * (1. - eps) + eps  
            z1 = torch.randn_like(x)
            alpha_t1, std1 = marginal_prob_std(random_t1, device = device)
            
            perturbed_x = x * alpha_t1[:, None] + z1 * std1[:, None]
            score1 = model(perturbed_x, random_t1)
            loss1 = torch.mean(torch.sum((score1 * std1[:, None] + z1)**2, dim=(1,)))*(1-gamma)
            model.condition = None
            ########################################################
            #target_timestep = torch.randint(low = 2, high = target_model.dpm_solver.noise_schedule.total_N -1, size = (1,))/target_model.dpm_solver.noise_schedule.total_N 
            if args.use_future_state:
                model.condition = conditions[0]
                target_model.condition = conditions[1]
                random_t2 = torch.rand(xf.shape[0], device=device) * (1. - eps) + eps 
                z2 = torch.randn_like(xf)
                alpha_t2, std2 = marginal_prob_std(random_t2, device = device)
        
                perturbed_xf = xf * alpha_t2[:, None] + z2 * std2[:, None]
                score2 = model(perturbed_xf, random_t2)  
                target_score2 = target_model(perturbed_xf, random_t2).detach()
                loss2 = torch.mean((1 - done.float())*torch.sum(((score2 * std2[:, None]) - (target_score2 * std2[:, None]))**2, dim=(1,)))*gamma
        
            else:
                model.condition = conditions[0]
                target_x = target_model.sample_occ(conditions[1], None).detach() 
                random_t2 = torch.rand(target_x.shape[0], device=device) * (1. - eps) + eps 
                z2 = torch.randn_like(target_x)
                alpha_t2, std2 = marginal_prob_std(random_t2, device = device)
            
                perturbed_x_target = target_x * alpha_t2[:, None] + z2 * std2[:, None]
                score2 = model(perturbed_x_target, random_t2)   
            
                loss2 = gamma*torch.mean(torch.sum((score2 * std2[:, None] + z2)**2, dim=(1,)))#gamma*torch.mean(torch.sum((1-done)*((target_score - score2)**2), dim=(1,)))
                
            loss = loss1 + loss2
        
    else:

            model.condition = conditions[0]
            random_t1 = torch.rand(xf.shape[0], device=device) * (1. - eps) + eps  
            z1 = torch.randn_like(xf)
            alpha_t1, std1 = marginal_prob_std(random_t1, device = device)
            
            perturbed_xf = xf * alpha_t1[:, None] + z1 * std1[:, None]
            score1 = model(perturbed_xf, random_t1)
            loss = torch.mean(torch.sum((score1 * std1[:, None] + z1)**2, dim=(1,)))
            loss1 = torch.zeros_like(loss)
            loss2 = torch.zeros_like(loss)
            
    model.condition = None
    target_model.condition = None
    return loss, loss1.clone().detach()/(1-gamma), loss2.clone().detach()/gamma

'''
def loss_fn_occ_new(args, model, target_model, x, xf, xf_, done, marginal_prob_std, eps=1e-3, gamma = 0.85, conditions = None, device = None):


    #sa, sa_ = conditions[0], conditions[1]
    model.condition = conditions[0]
    target_model.condition = conditions[1]
    
    random_t = torch.rand(x.shape[0], device=device) * (1. - eps) + eps  
    z1 = torch.randn_like(x)
    alpha_t, std = marginal_prob_std(random_t, device = device)

    #random_t2 = torch.rand(xf_.shape[0], device=device) * (1. - eps) + eps 
    z2 = torch.randn_like(xf_)
    #alpha_t2, std2 = marginal_prob_std(random_t2, device = device)

        
    perturbed_xf_ = xf_ * alpha_t[:, None] + z * std1[:, None]
    perturbed_xf = xf * alpha_t[:, None] + z * std2[:, None] #todo

    term1 =  -1*(z1 / std1[:, None])*(1-gamma)
    term2 = target_model(perturbed_xf_, random_t2).detach()*gamma
    target = term1 + term2 
 
    score = model(perturbed_xf, random_t1) #todo

    
    td_loss = torch.mean((1 - done.float())*torch.sum((target - score)**2, dim=(1,)))
   
            
    model.condition = None
    target_model.condition = None
    return td_loss, None, None'''