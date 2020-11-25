import torch


def generate_virtual_rollouts(model, agent, buffer, steps,
                              n_rollouts=1, term_fn=None,
                              stop_on_terminal=True, pessimism=0):

    model_is_training = model.training
    agent_is_training = agent.training

    model.eval()
    agent.eval()

    # The rollout consists of steps, where each step holds
    # [observation, action, reward, next_observation, done]
    out_observations = None
    out_actions = None
    out_next_observations = None
    out_rewards = None
    out_dones = None

    observations = buffer.sample_batch(n_rollouts)['obs']

    step = 0

    while step < steps and observations.numel() > 0:
        actions = agent.act(observations)
        pred = model.get_prediction(torch.as_tensor(
            torch.cat((observations, actions), dim=1),
            dtype=torch.float32), term_fn=term_fn,
            pessimism=pessimism)

        if out_observations is None:
            out_observations = observations.detach().clone()
            out_actions = actions.detach().clone()
            out_next_observations = pred[:, :-2].detach().clone()
            out_rewards = pred[:, -2].detach().clone()
            out_dones = pred[:, -1].detach().clone()
        else:
            out_observations = torch.cat((out_observations,
                                          observations.detach().clone()))
            out_actions = torch.cat((out_actions, actions.detach().clone()))
            out_next_observations = torch.cat((out_next_observations,
                                               pred[:, :-2].detach().clone()))
            out_rewards = torch.cat((out_rewards,
                                     pred[:, -2].detach().clone()))
            out_dones = torch.cat((out_dones, pred[:, -1].detach().clone()))

        if stop_on_terminal:
            observations = pred[pred[:, -1] == 0, :-2]
        else:
            observations = pred[:, :-2]

        step += 1

    if model_is_training:
        model.train()
    if agent_is_training:
        agent.train()

    return {'obs': out_observations,
            'act': out_actions,
            'rew': out_rewards,
            'next_obs': out_next_observations,
            'done': out_dones}


def generate_virtual_rollout(model, agent, start_observation, steps,
                             term_fn=None, stop_on_terminal=True):

    model_is_training = model.training
    agent_is_training = agent.training

    model.eval()
    agent.eval()

    # The rollout consists of steps, where each step holds
    # [observation, action, reward, next_observation, done]
    rollout = []

    this_observation = start_observation

    for step in range(steps):
        action = agent.act(this_observation)
        pred = model.get_prediction(torch.as_tensor(
            torch.cat((this_observation, action), dim=1),
            dtype=torch.float32), term_fn=term_fn)
        next_observation = pred[:, :-2]
        reward = pred[:, -2]
        done = pred[:, -1]

        rollout.append({'obs': this_observation, 'act': action,
                        'rew': reward, 'next_obs': next_observation,
                        'done': done})

        this_observation = next_observation

        if done and stop_on_terminal:
            break

    if model_is_training:
        model.train()
    if agent_is_training:
        agent.train()

    return rollout
