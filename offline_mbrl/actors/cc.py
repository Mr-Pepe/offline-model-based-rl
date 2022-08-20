import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from offline_mbrl.models.mlp import mlp
from offline_mbrl.models.mlp_q_function import MLPQFunction
from offline_mbrl.models.q_ensemble import QEnsemble
from offline_mbrl.models.squashed_gaussian_mlp_actor import \
    SquashedGaussianMLPActor
from offline_mbrl.utils.modes import UNDERESTIMATION
from torch.optim.adamw import AdamW


class CopyCat(nn.Module):
    def __init__(self, observation_space, action_space, hidden=(256, 256),
                 activation=nn.ReLU, pi_lr=3e-4, q_lr=3e-4, gamma=0.99,
                 alpha=0.2, polyak=0.995, batch_size=100, 
                 pre_fn=None, device='cpu', **_):

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.pre_fn = pre_fn
        self.device = device

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # TODO: Action limit for clamping: critically, assumes all dimensions
        # share the same bound!
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden, activation, act_limit)
        self.q = QEnsemble(obs_dim, act_dim)

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.q.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = AdamW(self.q_params, lr=q_lr)

        self.to(device)

        self.target = deepcopy(self)

        # Freeze target networks with respect to optimizers (only update via
        # polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], \
            data['rew'], data['obs2'], data['done']

        if self.pre_fn:
            o = self.pre_fn(o)
            o2 = self.pre_fn(o2)

        q = self.q(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.pi(o2)

            # Target Q-values
            q_pi_targ = self.target.q(o2, a2)
            q_pi_targ = q_pi_targ.min(dim=0).values
            backup = r + self.gamma * \
                ~d * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        q_info = dict(Q1Vals=q.cpu().detach().numpy(),
                      Q2Vals=0)

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']

        if self.pre_fn:
            o = self.pre_fn(o)

        pi, logp_pi = self.pi(o)
        q_pi = self.q(o, pi).min(dim=0).values

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, data):
        self.device = next(self.parameters()).device

        for key in data:
            data[key] = data[key].to(self.device)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to
                # update target params, as opposed to "mul" and "add", which
                # would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q, q_info, loss_pi, pi_info

    def multi_update(self, n_updates, buffer, model, logger=None, debug=False):
        for _ in range(n_updates):
            batch = buffer.sample_batch(self.batch_size)
            loss_q, q_info, loss_pi, pi_info = self.update(data=batch)

            if logger is not None:
                logger.store(LossQ=loss_q.item(), **q_info)
                logger.store(LossPi=loss_pi.item(), **pi_info)

    def act(self, o, deterministic=False):
        self.device = next(self.parameters()).device

        obs = torch.as_tensor(o,
                              dtype=torch.float32,
                              device=self.device)

        if self.pre_fn:
            obs = self.pre_fn(obs)

        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def act_randomly(self, o, deterministic=False):
        a = torch.as_tensor([self.action_space.sample() for _ in range(len(o))],
                            device=o.device)
        return a
