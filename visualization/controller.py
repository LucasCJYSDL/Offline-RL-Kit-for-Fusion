import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, TanhDiagGaussian, Actor

class Controller:
    def __init__(self, args):
        # register an actor
        actor_backbone = MLP(input_dim=args.obs_dim, hidden_dims=args.hidden_dims)
        if not args.stochastic_actor:
            self.actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
        else:
            dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=args.max_action
            )
            self.actor = ActorProb(actor_backbone, dist, args.device)
        
        # load up weights
        cwd = os.getcwd()
        actor_path = os.path.join(cwd, args.actor_path, 'checkpoint', 'policy.pth')
        checkpoint = torch.load(actor_path, map_location=args.device)

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("critic"):
                continue
            state_dict[k.replace("actor.", "")] = v 

        self.actor.load_state_dict(state_dict)
        self.actor.eval() # evaluation only

        # important arguments
        self.deterministic_mode = args.deterministic_mode
        self.stochastic_actor = args.stochastic_actor
    
    def act(self, obs):
        with torch.no_grad():
            if not self.stochastic_actor:
                action = self.actor(obs)
            else:
                dist = self.actor(obs)
                if self.deterministic_mode:
                    squashed_action, raw_action = dist.mode()
                else:
                    squashed_action, raw_action = dist.rsample()
                action = squashed_action

        return action.cpu().numpy()
