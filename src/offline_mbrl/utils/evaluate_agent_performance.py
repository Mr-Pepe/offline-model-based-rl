#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains a function to evaluate an agent in an environment."""

from typing import Union

import gym
import torch

from offline_mbrl.actors.behavioral_cloning import BehavioralCloningAgent
from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.actors.sac import SAC
from offline_mbrl.utils.logx import EpochLogger


def evaluate_agent_performance(
    test_env: gym.Env,
    agent: Union[RandomAgent, BehavioralCloningAgent, SAC],
    num_test_episodes: int,
    max_episode_length: int,
    logger: EpochLogger,
    render: bool = False,
) -> float:
    """Evaluates an agent's performance.

    Args:
        test_env (gym.Env): The environment to test the agent in.
        agent (Union[RandomAgent, BehavioralCloningAgent, SAC]): The agent to test.
        num_test_episodes (int): The number of episodes to run for evaluation.
        max_episode_length (int): The maximum number of steps in each episode.
        logger (EpochLogger): A logger to write test episode returns to.
        render (bool, optional): Whether or not to render the evaluation. Only renders
            the first test episode. Defaults to False.

    Returns:
        float: The average cumulated reward per episode.
    """
    total_return = 0.0

    for _ in range(num_test_episodes):
        done = False
        episode_return = 0.0
        episode_length = 0
        obs = torch.as_tensor(test_env.reset())

        while not (done or (episode_length == max_episode_length)):
            # Take deterministic actions at test time
            action = agent.act(obs, True).cpu().numpy()
            next_obs, reward, done, _ = test_env.step(action)
            if render:
                test_env.render()

            obs = next_obs

            episode_return += reward
            episode_length += 1

        logger.store(TestEpRet=episode_return, TestEpLen=episode_length)

        total_return += episode_return

        render = False

    return total_return / num_test_episodes
