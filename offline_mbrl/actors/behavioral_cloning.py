import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from offline_mbrl.models.mlp import mlp
from offline_mbrl.models.mlp_q_function import MLPQFunction
from offline_mbrl.models.squashed_gaussian_mlp_actor import SquashedGaussianMLPActor


class BC(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden=(200, 200, 200, 200),
        activation=nn.ReLU,
        lr=3e-4,
        batch_size=100,
        pre_fn=None,
        device="cpu",
        **_
    ):

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
        self.pi = mlp([obs_dim] + list(hidden) + [act_dim], activation)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=lr)

        self.to(device)

        self.use_amp = "cuda" in next(self.parameters()).device.type
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp, growth_factor=1.5, backoff_factor=0.7
        )

    def compute_loss_pi(self, data):
        o = data["obs"]
        a = data["act"]

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

        self.pi_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_pi = self.compute_loss_pi(data)

        self.scaler.scale(loss_pi).backward()
        self.scaler.step(self.pi_optimizer)
        self.scaler.update()

        return loss_pi

    def multi_update(self, n_updates, buffer, logger=None, debug=False):
        losses = torch.zeros(n_updates)
        for i_update in range(n_updates):
            batch = buffer.sample_batch(self.batch_size)
            loss_pi = self.update(data=batch)

            losses[i_update] = loss_pi

            if logger is not None:
                logger.store(LossQ=0, Q1Vals=0, Q2Vals=0)
                logger.store(LossPi=loss_pi.item(), LogPi=0)

        if debug:
            return losses.mean()

    def act(self, o, deterministic=False):
        self.device = next(self.parameters()).device

        obs = torch.as_tensor(o, dtype=torch.float32, device=self.device)

        if self.pre_fn:
            obs = self.pre_fn(obs)

        with torch.no_grad():
            return self.pi(obs)

    def act_randomly(self, o, deterministic=False):
        a = torch.as_tensor(
            [self.action_space.sample() for _ in range(len(o))], device=o.device
        )
        return a
