#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains an implementation of a behavioral cloning agent."""

from typing import Callable, Optional, Type

import torch
from gym.spaces import Space
from torch import nn
from torch.optim.adamw import AdamW

from offline_mbrl.models.mlp import mlp
from offline_mbrl.utils.logx import EpochLogger
from offline_mbrl.utils.replay_buffer import ReplayBuffer


class BC(nn.Module):
    # Disable pylint error for not implementing forward method
    # pylint: disable=abstract-method
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        hidden_layer_sizes: tuple[int, ...] = (200, 200, 200, 200),
        activation: Type[nn.Module] = nn.ReLU,
        lr: float = 3e-4,
        batch_size: int = 100,
        preprocessing_function: Callable = None,
        device: str = "cpu",
        **_: dict
    ):
        super().__init__()

        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.device = device

        assert observation_space.shape is not None
        assert action_space.shape is not None

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # build policy and value functions
        self.pi = mlp([obs_dim] + list(hidden_layer_sizes) + [act_dim], activation)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=lr)

        self.to(device)

        self.use_amp = "cuda" in next(self.parameters()).device.type
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp, growth_factor=1.5, backoff_factor=0.7
        )

    def compute_loss_pi(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        o = data["obs"]
        a = data["act"]

        if self.preprocessing_function:
            o = self.preprocessing_function(o)

        criterion = nn.MSELoss()

        pi = self.pi(o)

        loss_pi = criterion(pi, a)

        return loss_pi

    def update(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        self.device = next(self.parameters()).device.type

        for key in data:
            data[key] = data[key].to(self.device)

        self.pi_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_pi = self.compute_loss_pi(data)

        self.scaler.scale(loss_pi).backward()
        self.scaler.step(self.pi_optimizer)
        self.scaler.update()

        return loss_pi

    def multi_update(
        self,
        n_updates: int,
        buffer: ReplayBuffer,
        logger: EpochLogger = None,
        debug: bool = False,
    ) -> Optional[torch.Tensor]:
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

        return None

    def act(
        self, observation: torch.Tensor, unused_deterministic: bool = True
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        self.device = device.type

        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)

        if self.preprocessing_function:
            obs = self.preprocessing_function(obs)

        with torch.no_grad():
            return self.pi(obs)

    def act_randomly(
        self, observation: torch.Tensor, unused_deterministic: bool = False
    ) -> torch.Tensor:
        action = torch.as_tensor(
            [self.action_space.sample() for _ in range(len(observation))],
            device=observation.device,
        )
        return action
