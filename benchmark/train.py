from benchmark.utils.rollout_length_from_schedule import \
    get_rollout_length_from_schedule
from benchmark.utils.virtual_rollouts import generate_virtual_rollout
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.evaluate_policy import test_agent
import time

import numpy as np
import torch

from benchmark.actors.sac import SAC
from benchmark.utils.logx import EpochLogger
from benchmark.utils.replay_buffer import ReplayBuffer


class Trainer():

    def __init__(self,
                 env_fn,
                 term_fn=None,
                 sac_kwargs=dict(),
                 model_kwargs=dict(),
                 seed=0,
                 epochs=100,
                 steps_per_epoch=4000,
                 random_steps=10000,
                 init_steps=1000,
                 agent_updates_per_step=1,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 use_model=False,
                 rollouts_per_step=10,
                 rollout_schedule=[1, 1, 20, 100],
                 train_model_every=250,
                 replay_size=int(1e6),
                 logger_kwargs=dict(),
                 save_freq=1,
                 device='cpu',
                 render=False):
        """

        Args:
            term_fn (str): A termination function as specified in
                termination_functions.py. If None, the termination function will
                be learned with an extra network.

            epochs (int): Number of epochs to run and train agent.

            steps_per_epoch (int): Number of steps of interaction (state-action
                pairs) for the agent and the environment in each epoch.

            replay_size (int): Maximum length of replay buffer.

            random_steps (int): Number of steps for uniform-random action
            selection, before running real policy. Helps exploration.

            init_steps (int): Number of env interactions to collect before
                starting to do gradient descent updates or training an
                environment model. Ensures replay buffer is full enough for
                useful updates.

            num_test_episodes (int): Number of episodes to test the
            deterministic policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            use_model (bool): Whether to augment data with virtual rollouts.

            model_rollouts (int): The number of model rollouts to perform per
                environment step.

            train_model_every (int): After how many steps the model should be
                retrained.

            agent_updates_per_step (int): The number of agent updates to
                perform per environment step.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Create SAC and environment model
        sac_kwargs.update({'device': device})
        model_kwargs.update({'device': device})
        self.sac_kwargs = sac_kwargs
        self.model_kwargs = model_kwargs
        self.agent = SAC(self.env.observation_space,
                         self.env.action_space,
                         **sac_kwargs)
        self.env_model = EnvironmentModel(obs_dim[0],
                                          act_dim,
                                          **model_kwargs)
        self.env_model.to(device)

        self.logger.setup_pytorch_saver(self.agent)

        self.real_replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                               act_dim=act_dim,
                                               size=replay_size,
                                               device=device)

        self.virtual_replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                                  act_dim=act_dim,
                                                  size=replay_size,
                                                  device=device)

        self.term_fn = term_fn
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_steps = init_steps
        self.random_steps = random_steps
        self.total_steps = steps_per_epoch * epochs
        self.max_ep_len = min(max_ep_len, self.total_steps)

        self.agent_updates_per_step = agent_updates_per_step

        self.use_model = use_model
        self.rollouts_per_step = rollouts_per_step
        self.rollout_schedule = rollout_schedule
        self.train_model_every = train_model_every

        self.num_test_episodes = num_test_episodes
        self.save_freq = save_freq
        self.render = render

    def train(self):

        final_return = None

        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        if self.total_steps < self.init_steps:
            raise ValueError(
                """Number of total steps too low. Increase number of epochs or
                steps per epoch.""")

        step_total = 0
        steps_since_model_training = 1e10

        for epoch in range(1, self.epochs + 1):

            agent_update_performed = False
            model_trained = False

            rollout_length = get_rollout_length_from_schedule(
                self.rollout_schedule,
                epoch)

            print("Epoch {}\tRollout length: {}".format(epoch, rollout_length))

            # Main loop: collect experience in env and update/log each epoch
            for step_epoch in range(self.steps_per_epoch):

                # Train environment model on real experience
                if self.use_model and \
                        self.real_replay_buffer.size > 0 and \
                        step_total > self.init_steps and \
                        steps_since_model_training >= self.train_model_every:

                    model_val_error = self.env_model.train_to_convergence(
                        self.real_replay_buffer,
                        self.model_kwargs)

                    model_trained = True
                    steps_since_model_training = 0
                    self.logger.store(LossEnvModel=model_val_error)
                    print('')
                    print('Environment model error: {}'.format(model_val_error))

                print("Epoch {}, step {}/{}".format(epoch,
                                                    step_epoch+1,
                                                    self.steps_per_epoch),
                      end='\r')

                if step_total > self.random_steps:
                    a = self.agent.get_action(o)
                else:
                    a = self.env.action_space.sample()

                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d = False if ep_len == self.max_ep_len else d

                self.real_replay_buffer.store(o, a, r, o2, d)
                o = o2

                # End of trajectory handling
                if d or (ep_len == self.max_ep_len):
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

                # Update agent
                if step_total >= self.init_steps:
                    agent_update_performed = True

                    if self.use_model:
                        for model_rollout in range(self.model_rollouts):
                            start_observation = self \
                                .real_replay_buffer \
                                .sample_batch(1)['obs']

                            rollout = generate_virtual_rollout(
                                self.env_model,
                                self.agent,
                                start_observation,
                                rollout_length,
                                term_fn=self.term_fn)
                            for step in rollout:
                                self.virtual_replay_buffer.store(
                                    step['o'], step['act'], step['rew'],
                                    step['o2'], step['d'])

                        self.agent.multi_update(self.agent_updates_per_step,
                                                self.virtual_replay_buffer,
                                                self.logger)
                    else:
                        self.agent.multi_update(self.agent_updates_per_step,
                                                self.real_replay_buffer,
                                                self.logger)

                steps_since_model_training += 1

                step_total += 1

            print('')

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                self.logger.save_state({'env': self.env}, None)

            # Test the performance of the deterministic version of the agent.
            final_return = test_agent(self.test_env,
                                      self.agent,
                                      self.max_ep_len,
                                      self.num_test_episodes,
                                      self.logger,
                                      self.render)

            log_end_of_epoch(self.logger, epoch, step_total, start_time,
                             agent_update_performed, model_trained,
                             rollout_length)

        return final_return


def log_end_of_epoch(logger, epoch, step_total, start_time,
                     agent_update_performed, model_trained, rollout_length):

    logger.log_tabular('Epoch', epoch)

    logger.log_tabular('RolloutLength', rollout_length)

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
