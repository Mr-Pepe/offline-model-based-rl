#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains a function to create simulated training data for RL agents."""


from typing import Optional, Union, cast

import torch

from offline_mbrl.actors.behavioral_cloning import BC
from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.actors.sac import SAC
from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.utils.replay_buffer import ReplayBuffer


def generate_virtual_rollouts(
    model: EnvironmentModel,
    agent: Union[BC, SAC],
    buffer: ReplayBuffer,
    steps: int,
    n_rollouts: int = 1,
    pessimism: float = 0,
    ood_threshold: float = -1,
    random_action: bool = False,
    prev_obs: Optional[dict] = None,
    max_rollout_length: int = -1,
    mode: Optional[str] = None,
) -> tuple[dict, dict]:
    """Creates virtual rollouts, given an environment model and an agent.

    Previously started rollouts can be continued by passing the :code:`prev_obs`
    argument.

    Args:
        model (EnvironmentModel): The model to use for predicting interaction results.
        agent (Union[RandomAgent, BC, SAC]): The agent to use for interacting with the
            environment model.
        buffer (ReplayBuffer): The replay buffer to sample starting states from.
        steps (int): The number of steps to generate for each rollout.
        n_rollouts (int, optional): The number of rollouts to generate in parallel.
            Defaults to 1.
        pessimism (float, optional): The pessimism coefficient to pass to the
            environment model. Defaults to 0.
        ood_threshold (float, optional): The out-of-distribution threshold to pass to
            the environment model. Defaults to -1.
        random_action (bool, optional): Whether or not to use random actions, in which
            case the agent must provide an :code:`act_randomly` method. Defaults to
            False.
        prev_obs (Optional[dict], optional): A dictionary containing previous
            observations and the length of their respective rollouts. Defaults to None.
        max_rollout_length (int, optional): After how many steps rollouts should be
            terminated if they did not encounter a terminal state earlier. Defaults to
            -1.
        mode (Optional[str], optional): The mode to pass to the environment model.
            Defaults to None.

    Returns:
        tuple[dict, dict]: A dictionary containing the virtual rollouts and a
            dictionary that can be passed as the :code:`prev_obs` argument to the next
            call of this function to continue rollouts.
    """
    # Remember whether the model or agent was in training mode
    model_was_training = model.training
    agent_was_training = agent.training

    model.eval()
    agent.eval()

    # The rollout consists of steps, where each step holds
    # [observation, action, reward, next_observation, done]
    out_observations = None
    out_actions = None
    out_next_observations = None
    out_rewards = None
    out_dones = None

    if prev_obs is None:
        observations = buffer.sample_batch(n_rollouts, non_terminals_only=True)["obs"]
        lengths = torch.zeros((n_rollouts)).to(observations.device)
    else:
        observations = prev_obs["obs"]
        lengths = prev_obs["lengths"]

        n_new_rollouts = n_rollouts - len(observations)

        if len(observations) < n_rollouts:
            observations = torch.cat(
                (observations, buffer.sample_batch(n_new_rollouts, True)["obs"])
            )

            lengths = torch.cat(
                (lengths, torch.zeros(n_new_rollouts).to(lengths.device))
            )

    step = 0

    while step < steps and observations.numel() > 0:
        if random_action:
            actions = agent.act_randomly(observations)
        else:
            actions = agent.act(observations)

        pred = cast(
            torch.Tensor,
            model.get_prediction(
                torch.as_tensor(
                    torch.cat((observations, actions), dim=1), dtype=torch.float32
                ),
                pessimism=pessimism,
                ood_threshold=ood_threshold,
                mode=mode,
            ),
        )

        observations = observations.detach().clone()
        actions = actions.detach().clone()
        next_observations = pred[:, :-2].detach().clone()
        rewards = pred[:, -2].detach().clone()
        dones = pred[:, -1].detach().clone()

        if out_observations is None:
            out_observations = observations
            out_actions = actions
            out_next_observations = next_observations
            out_rewards = rewards
            out_dones = dones
        else:
            out_observations = torch.cat((out_observations, observations))
            out_actions = torch.cat((out_actions, actions))
            out_next_observations = torch.cat(
                (out_next_observations, next_observations)
            )
            out_rewards = torch.cat((out_rewards, rewards))
            out_dones = torch.cat((out_dones, dones))

        lengths += 1

        if max_rollout_length != -1:
            # Terminate rollouts that have reached the maximum length
            dones = torch.logical_or(dones, lengths == max_rollout_length)

        observations = next_observations[dones == 0]
        lengths = lengths[dones == 0]

        step += 1

    if model_was_training:
        model.train()
    if agent_was_training:
        agent.train()

    return {
        "obs": out_observations,
        "act": out_actions,
        "rew": out_rewards,
        "next_obs": out_next_observations,
        "done": out_dones,
    }, {"obs": observations, "lengths": lengths}
