from benchmark.utils.evaluate_policy import test_agent
import time
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from benchmark.actors.sac import SAC
from benchmark.utils.count_vars import count_vars
from benchmark.utils.logx import EpochLogger
from benchmark.utils.replay_buffer import ReplayBuffer


def train(env_fn, sac_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
          batch_size=100, start_steps=10000,
          update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
          logger_kwargs=dict(), save_freq=1, device='cpu'):
    """

    Args:
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # Based on https://spinningup.openai.com

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    final_return = None

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Create actor-critic module and target networks
    sac_kwargs.update({'device': device})
    agent = SAC(env.observation_space, env.action_space, **sac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    var_counts = tuple(count_vars(module)
                       for module in [agent.pi, agent.q1, agent.q2])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up model saving
    # TODO: Save environment model
    logger.setup_pytorch_saver(agent)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    max_ep_len = min(max_ep_len, total_steps)

    if total_steps < update_after + ((total_steps - update_after) % update_every):
        raise ValueError(
            'Number of total steps too low. Increase number of epochs or steps per epoch.')

    step_total = 0

    for epoch in range(epochs):

        # Main loop: collect experience in env and update/log each epoch
        for step_epoch in range(steps_per_epoch):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if step_total > start_steps:
                a = agent.get_action(o)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if step_total >= update_after and step_total % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    loss_q, q_info, loss_pi, pi_info = agent.update(data=batch)
                    logger.store(LossQ=loss_q.item(), **q_info)
                    logger.store(LossPi=loss_pi.item(), **pi_info)

            step_total += 1

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        final_return = test_agent(
            test_env, agent, max_ep_len, num_test_episodes, logger)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', step_total)
        logger.log_tabular('Q1Vals', with_min_and_max=True)
        logger.log_tabular('Q2Vals', with_min_and_max=True)
        logger.log_tabular('LogPi', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    return final_return
