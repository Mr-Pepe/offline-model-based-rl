from benchmark.models.mlp import mlp
import itertools

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from benchmark.models.mlp_q_function import MLPQFunction
from benchmark.models.squashed_gaussian_mlp_actor import \
    SquashedGaussianMLPActor


class BC(nn.Module):
    def __init__(self, observation_space, action_space, hidden=(64, 64, 64, 64),
                 activation=nn.ReLU, lr=3e-4, batch_size=100,
                 pre_fn=None, device='cpu', **_):

        super().__init__()

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
        self.pi = mlp([obs_dim] + list(hidden) + [act_dim], activation, nn.Tanh)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=lr)

        self.to(device)

    def compute_loss_pi(self, data):
        o = data['obs']
        a = data['act']

        if self.pre_fn:
            o = self.pre_fn(o)

        criterion = nn.MSELoss()

        pi = self.pi(o)

        loss_pi = criterion(pi, a)

        return loss_pi

    def update(self, data):
        self.device = next(self.parameters()).device

        for key in data:
            data[key] = data[key].to(self.device)

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        return loss_pi

    def multi_update(self, n_updates, buffer, logger=None):
        for _ in range(n_updates):
            batch = buffer.sample_batch(self.batch_size)
            loss_pi = self.update(data=batch)

            if logger is not None:
                logger.store(LossQ=0, Q1Vals=0, Q2Vals=0)
                logger.store(LossPi=loss_pi.item(), LogPi=0)

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
