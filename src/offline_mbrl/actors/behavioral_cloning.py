#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains an implementation of a behavioral cloning agent."""

from typing import Optional

import torch
from gym.spaces import Space
from torch import nn
from torch.optim.adamw import AdamW

from offline_mbrl.models.mlp import mlp
from offline_mbrl.schemas import BehavioralCloningConfiguration
from offline_mbrl.utils.logx import EpochLogger
from offline_mbrl.utils.replay_buffer import ReplayBuffer


class BehavioralCloningAgent(nn.Module):
    # Disable pylint error for not implementing forward method
    # pylint: disable=abstract-method
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: BehavioralCloningConfiguration = BehavioralCloningConfiguration(),
    ):
        super().__init__()

        self.config = config

        assert observation_space.shape is not None
        assert action_space.shape is not None

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # build policy and value functions
        self.pi = mlp((obs_dim, *config.hidden_layer_sizes, act_dim), config.activation)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=self.config.lr)

        self.to(config.device)

        self.use_amp = "cuda" in next(self.parameters()).device.type
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp, growth_factor=1.5, backoff_factor=0.7
        )

    def compute_loss_pi(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        o = data["obs"]
        a = data["act"]

        if self.config.preprocessing_function:
            o = self.config.preprocessing_function(o)

        criterion = nn.MSELoss()

        pi = self.pi(o)

        loss_pi = criterion(pi, a)

        return loss_pi

    def update(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        self.config.device = next(self.parameters()).device.type

        for key in data:
            data[key] = data[key].to(self.config.device)

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
            batch = buffer.sample_batch(self.config.training_batch_size)
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
        self.config.device = device.type

        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)

        if self.config.preprocessing_function:
            obs = self.config.preprocessing_function(obs)

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
