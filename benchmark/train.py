from benchmark.utils.virtual_rollouts import generate_virtual_rollout
from benchmark.utils.train_environment_model import train_environment_model
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.evaluate_policy import test_agent
import time

import numpy as np
import torch

from benchmark.actors.sac import SAC
from benchmark.utils.count_vars import count_vars
from benchmark.utils.logx import EpochLogger
from benchmark.utils.replay_buffer import ReplayBuffer


def train(env_fn, sac_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
          batch_size=100, random_steps=10000,
          init_steps=1000, num_test_episodes=10, max_ep_len=1000,
          use_model=False, model_type='deterministic', n_networks=1,
          model_rollouts=10, train_model_every=250,
          model_batch_size=128, model_lr=1e-3, model_val_split=0.2,
          agent_updates=1,
          logger_kwargs=dict(), save_freq=1, device='cpu'):
    """

    Args:
        epochs (int): Number of epochs to run and train agent.

        steps_per_epoch (int): Number of steps of interaction (state-action
            pairs) for the agent and the environment in each epoch.

        replay_size (int): Maximum length of replay buffer.

        batch_size (int): Minibatch size for SGD.

        random_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        init_steps (int): Number of env interactions to collect before
            starting to do gradient descent updates or training an environment
            model. Ensures replay buffer is full enough for useful updates.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        use_model (bool): Whether to augment data with virtual rollouts.

        model_type (string): Environment model type:
            deterministic or probabilistic

        n_networks (int): The number of networks to use as an ensemble.

        model_rollouts (int): The number of model rollouts to perform per
            environment step.

        train_model_every (int): After how many steps the model should be
            retrained.

        model_batch_size (int): Batch size for training the environment model.

        model_lr (float): Learning rate for training the environment model.

        model_val_split (float): Fraction of data to use as validation set for
            training of environment model.

        agent_updates (int): The number of agent updates to perform per
            environment step.

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

    # Create SAC and environment model
    sac_kwargs.update({'device': device})
    agent = SAC(env.observation_space, env.action_space, **sac_kwargs)
    env_model = EnvironmentModel(obs_dim[0],
                                 act_dim,
                                 type=model_type,
                                 n_networks=n_networks)
    env_model.to(device)

    # TODO: Save environment model
    logger.setup_pytorch_saver(agent)

    real_replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    var_counts = tuple(count_vars(module)
                       for module in [agent.pi, agent.q1, agent.q2])
    logger.log(
        '\nNumber of parameters: \t pi: {}, \t q1: {}, \t q2: {}\n'
        .format(*var_counts))

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    max_ep_len = min(max_ep_len, total_steps)

    if total_steps < init_steps:
        raise ValueError(
            """Number of total steps too low. Increase number of epochs or
            steps per epoch.""")

    step_total = 0

    for epoch in range(epochs):

        agent_update_performed = False
        model_trained = False

        # Main loop: collect experience in env and update/log each epoch
        for step_epoch in range(steps_per_epoch):

            # Train environment model on real experience
            if use_model and \
                    real_replay_buffer.size > 0 and \
                    step_total > init_steps and \
                    step_total % train_model_every == 0:
                model_val_error = train_environment_model(
                    env_model, real_replay_buffer, model_lr, model_batch_size,
                    model_val_split)
                model_trained = True
                logger.store(LossEnvModel=model_val_error)
                print('')
                print('Environment model error: {}'.format(model_val_error))

            print("Epoch {}, step {}/{}".format(epoch,
                                                step_epoch+1,
                                                steps_per_epoch), end='\r')

            if step_total > random_steps:
                a = agent.get_action(o)
            else:
                a = env.action_space.sample()

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            real_replay_buffer.store(o, a, r, o2, d)
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update agent
            if step_total >= init_steps:
                agent_update_performed = True

                if use_model:
                    # TODO: Adapt to rollout length schedule
                    virtual_replay_buffer = ReplayBuffer(
                        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
                    for model_rollout in range(model_rollouts):
                        start_observation = real_replay_buffer.sample_batch(1)[
                            'obs']
                        rollout = generate_virtual_rollout(
                            env_model, agent, start_observation, 1)
                        for step in rollout:
                            virtual_replay_buffer.store(
                                step['o'], step['act'], step['rew'],
                                step['o2'], step['d'])

                    update_agent(agent, agent_updates,
                                 virtual_replay_buffer, batch_size, logger)
                else:
                    # Update regular SAC
                    update_agent(agent, agent_updates,
                                 real_replay_buffer, batch_size, logger)

            step_total += 1

        print('')

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        final_return = test_agent(
            test_env, agent, max_ep_len, num_test_episodes, logger)

        log_end_of_epoch(logger, epoch, step_total, start_time,
                         agent_update_performed, model_trained)

    return final_return


def update_agent(agent, n_updates, buffer, batch_size, logger):
    for j in range(n_updates):
        batch = buffer.sample_batch(batch_size)
        loss_q, q_info, loss_pi, pi_info = agent.update(data=batch)
        logger.store(LossQ=loss_q.item(), **q_info)
        logger.store(LossPi=loss_pi.item(), **pi_info)


def log_end_of_epoch(logger, epoch, step_total, start_time,
                     agent_update_performed, model_trained):

    logger.log_tabular('Epoch', epoch)

    logger.log_tabular('EpRet', with_min_and_max=True)

    logger.log_tabular('TestEpRet', with_min_and_max=True)

    logger.log_tabular('EpLen', average_only=True)

    logger.log_tabular('TestEpLen', average_only=True)

    logger.log_tabular('TotalEnvInteracts', step_total)

    # Use placeholder value if no model update has been performed yet
    if not model_trained:
        logger.store(LossEnvModel=0)

    logger.log_tabular('LossEnvModel', with_min_and_max=True)

    # Use placeholder values if no agent update has been performed yet
    if not agent_update_performed:
        logger.store(Q1Vals=0,
                     Q2Vals=0,
                     LogPi=0,
                     LossPi=0,
                     LossQ=0)

    logger.log_tabular('Q1Vals', with_min_and_max=True)
    logger.log_tabular('Q2Vals', with_min_and_max=True)
    logger.log_tabular('LogPi', with_min_and_max=True)
    logger.log_tabular('LossPi', average_only=True)
    logger.log_tabular('LossQ', average_only=True)

    logger.log_tabular('Time', time.time()-start_time)
    logger.dump_tabular()
