import torch


def generate_virtual_rollouts(model, agent, buffer, steps,
                              n_rollouts=1, stop_on_terminal=True, pessimism=0,
                              random_action=False, prev_obs=None,
                              max_rollout_length=-1, exploration_mode='state'):

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

    if prev_obs is None:
        batch = buffer.sample_batch(n_rollouts)
        dones = model.term_fn(next_obs=batch['obs'].unsqueeze(0)).view(-1)
        observations = batch['obs'][dones == 0]
        lengths = torch.zeros(len(observations)).to(observations.device)
    else:
        observations = prev_obs['obs']
        lengths = prev_obs['lengths']

    while len(observations) < n_rollouts:
        n_new_rollouts = n_rollouts-len(observations)

        batch = buffer.sample_batch(n_new_rollouts)
        dones = model.term_fn(next_obs=batch['obs'].unsqueeze(0)).view(-1)
        new_obs = batch['obs'][dones == 0]

        observations = torch.cat((
            observations,
            new_obs
        ))

        lengths = torch.cat(
            (lengths, torch.zeros(len(new_obs)).to(lengths.device)))

    step = 0

    while step < steps and observations.numel() > 0:
        if random_action:
            actions = agent.act_randomly(observations)
        else:
            actions = agent.act(observations)

        pred = model.get_prediction(
            torch.as_tensor(torch.cat((observations,
                                       actions), dim=1),
                            dtype=torch.float32),
            pessimism=pessimism,
            exploration_mode=exploration_mode)

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
            out_observations = torch.cat((out_observations,
                                          observations))
            out_actions = torch.cat((out_actions, actions))
            out_next_observations = torch.cat((out_next_observations,
                                               next_observations))
            out_rewards = torch.cat((out_rewards, rewards))
            out_dones = torch.cat((out_dones, dones))

        lengths += 1

        if stop_on_terminal:
            if max_rollout_length != -1:
                dones = torch.logical_or(dones, lengths == max_rollout_length)
            observations = next_observations[dones == 0]
            lengths = lengths[dones == 0]
        else:
            observations = next_observations

        step += 1

    if model_is_training:
        model.train()
    if agent_is_training:
        agent.train()

    return {'obs': out_observations,
            'act': out_actions,
            'rew': out_rewards,
            'next_obs': out_next_observations,
            'done': out_dones}, {'obs': observations, 'lengths': lengths}


def generate_virtual_rollout(model, agent, start_observation, steps,
                             stop_on_terminal=True):

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
            dtype=torch.float32))
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
