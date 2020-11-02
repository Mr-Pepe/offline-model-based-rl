import numpy as np
import torch


def generate_virtual_rollout(model, agent, start_observation, steps,
                             term_fn=None):

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
            np.concatenate((this_observation, action), axis=1),
            dtype=torch.float32), term_fn=term_fn)
        next_observation = pred[:, :-2]
        reward = pred[:, -2]
        done = pred[:, -1]

        rollout.append({'o': this_observation, 'act': action,
                        'rew': reward, 'o2': next_observation, 'd': done})

        this_observation = next_observation

        if done:
            break

    if model_is_training:
        model.train()
    if agent_is_training:
        agent.train()

    return rollout
