import numpy as np
import torch
from torch._C import dtype


def generate_virtual_rollout(model, agent, start_observation, steps):
    model_is_training = model.training
    agent_is_training = agent.training

    model.eval()
    agent.eval()

    # The rollout consists of steps, where each step holds
    # [observation, action, reward, next_observation, done]
    rollout = []

    this_observation = start_observation

    for step in range(steps):
        action = agent.get_action(this_observation)
        pred = model.get_prediction(torch.as_tensor(
            np.concatenate((this_observation, action)), dtype=torch.float32))
        next_observation = pred[0, :-1]
        reward = pred[0, -1]

        rollout.append([this_observation, action,
                        reward, next_observation, False])

        this_observation = next_observation

    if model_is_training:
        model.train()
    if agent_is_training:
        agent.train()

    return rollout
