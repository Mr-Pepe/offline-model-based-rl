from benchmark.utils.termination_functions import get_termination_function
import gym
from benchmark.utils.model_needs_training import model_needs_training
from benchmark.utils.actions import Actions
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.rollout_length_from_schedule import \
    get_rollout_length_from_schedule
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.evaluate_policy import test_agent
import time

import torch
import numpy as np

from benchmark.actors.sac import SAC
from benchmark.utils.logx import EpochLogger
from benchmark.utils.replay_buffer import ReplayBuffer


class Trainer():

    def __init__(self,
                 env_name,
                 sac_kwargs=dict(),
                 model_kwargs=dict(),
                 seed=0,
                 epochs=100,
                 steps_per_epoch=4000,
                 random_steps=10000,
                 init_steps=1000,
                 n_samples_from_dataset=-1,
                 agent_updates_per_step=1,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 use_model=False,
                 model_pessimism=0,
                 model_max_n_train_batches=-1,
                 rollouts_per_step=10,
                 rollout_schedule=[1, 1, 20, 100],
                 max_rollout_length=999999,
                 continuous_rollouts=False,
                 train_model_every=250,
                 real_buffer_size=int(1e6),
                 virtual_buffer_size=int(1e6),
                 pretrain_epochs=0,
                 logger_kwargs=dict(),
                 save_freq=1,
                 device='cpu',
                 render=False):
        """

        Args:
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

            rollouts_per_step (int): The number of model rollouts to perform per
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

        logger_kwargs.update({'env_name': env_name})
        self.logger = EpochLogger(**logger_kwargs)
        local_vars = locals()
        self.logger.save_config(
            {key: local_vars[key] for key in local_vars if key != 'self'})

        self.env, self.test_env = gym.make(env_name), gym.make(env_name)
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.test_env.seed(seed)
        self.test_env.action_space.seed(seed)

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

        if pretrain_epochs > 0:
            self.real_replay_buffer, _, _ = load_dataset_from_env(
                self.env,
                n_samples=n_samples_from_dataset,
                buffer_size=real_buffer_size,
                buffer_device=device)
        else:
            self.real_replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                                   act_dim=act_dim,
                                                   size=real_buffer_size,
                                                   device=device)

        self.virtual_replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                                  act_dim=act_dim,
                                                  size=virtual_buffer_size,
                                                  device=device)

        self.logger.setup_pytorch_saver({
            'agent': self.agent,
            'model': self.env_model,
            'replay_buffer': self.real_replay_buffer,
            'virtual_replay_buffer': self.virtual_replay_buffer,
        }
        )

        # Get termination for environment, if model should be used
        if use_model:
            self.term_fn = get_termination_function(env_name)
            if not self.term_fn:
                raise ValueError("Could not find termination function for \
            environment {}".format(env_name))
        else:
            self.term_fn = None

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_steps = init_steps
        self.random_steps = random_steps
        self.total_steps = steps_per_epoch * epochs + \
            steps_per_epoch * pretrain_epochs
        self.max_ep_len = min(max_ep_len, self.total_steps)
        self.pretrain_epochs = pretrain_epochs

        self.agent_updates_per_step = agent_updates_per_step

        self.use_model = use_model
        self.rollouts_per_step = rollouts_per_step
        self.rollout_schedule = rollout_schedule
        self.max_rollout_length = max_rollout_length
        self.train_model_every = train_model_every
        self.continuous_rollouts = continuous_rollouts
        self.model_pessimism = model_pessimism
        self.model_max_n_train_batches = model_max_n_train_batches

        self.num_test_episodes = num_test_episodes
        self.save_freq = save_freq
        self.render = render

    def train(self):

        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        if self.total_steps < self.init_steps:
            raise ValueError(
                """Number of total steps lower than init steps.
                Increase number of epochs or steps per epoch.""")

        step_total = -self.pretrain_epochs*self.steps_per_epoch
        steps_since_model_training = 1e10

        test_performances = []
        action_log = []

        model_trained_at_all = False

        prev_obs = None

        for epoch in range(-self.pretrain_epochs+1, self.epochs + 1):

            agent_update_performed = False
            model_trained_this_epoch = False
            episode_finished = False
            tested_agent = False

            rollout_length = get_rollout_length_from_schedule(
                self.rollout_schedule,
                epoch)

            print("Epoch {}\tRollout length: {}".format(epoch, rollout_length))

            for step_epoch in range(self.steps_per_epoch):
                actions_this_step = [0 for i in range(len(Actions))]

                take_random_action = \
                    step_total + self.pretrain_epochs*self.steps_per_epoch < \
                    self.random_steps

                # Train environment model on real experience
                if model_needs_training(step_total,
                                        self.use_model,
                                        self.real_replay_buffer.size,
                                        self.init_steps,
                                        steps_since_model_training,
                                        self.train_model_every,
                                        model_trained_at_all):

                    model_val_error, _ = self.env_model.train_to_convergence(
                        self.real_replay_buffer,
                        patience_value=0 if epoch < 1 else 1,
                        max_n_train_batches=-1 if epoch < 1 else self.model_max_n_train_batches,
                        **self.model_kwargs)

                    model_val_error = model_val_error.mean()
                    model_trained_at_all = True
                    model_trained_this_epoch = True
                    steps_since_model_training = 0
                    self.logger.store(LossEnvModel=model_val_error)
                    actions_this_step[Actions.TRAIN_MODEL] = 1
                    print('')

                print("Epoch {}, step {}/{}".format(epoch,
                                                    step_epoch+1,
                                                    self.steps_per_epoch),
                      end='\r')

                if epoch > 0:
                    if take_random_action:
                        a = self.env.action_space.sample()
                        actions_this_step[Actions.RANDOM_ACTION] = 1
                    else:
                        a = self.agent.act(o).cpu().numpy()

                    o2, r, d, _ = self.env.step(a)
                    ep_ret += r
                    ep_len += 1

                    # Ignore the "done" signal if it comes from hitting the time
                    # horizon (that is, when it's an artificial terminal signal
                    # that isn't based on the agent's state)
                    d = False if ep_len == self.max_ep_len else d

                    self.real_replay_buffer.store(torch.as_tensor(o),
                                                  torch.as_tensor(a),
                                                  torch.as_tensor(r),
                                                  torch.as_tensor(o2),
                                                  torch.as_tensor(d))
                    o = o2

                    # End of trajectory handling
                    if d or (ep_len == self.max_ep_len):
                        episode_finished = True
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        o, ep_ret, ep_len = self.env.reset(), 0, 0

                    steps_since_model_training += 1
                    actions_this_step[Actions.INTERACT_WITH_ENV] = 1

                # Update agent
                if step_total >= self.init_steps or self.pretrain_epochs > 0:
                    if self.use_model:
                        rollouts, prev_obs = generate_virtual_rollouts(
                            self.env_model,
                            self.agent,
                            self.real_replay_buffer,
                            rollout_length,
                            n_rollouts=self.rollouts_per_step,
                            term_fn=self.term_fn,
                            pessimism=self.model_pessimism,
                            random_action=take_random_action,
                            prev_obs=prev_obs if self.continuous_rollouts else None,
                            max_rollout_length=self.max_rollout_length
                        )
                        self.virtual_replay_buffer.store_batch(
                            rollouts['obs'],
                            rollouts['act'],
                            rollouts['rew'],
                            rollouts['next_obs'],
                            rollouts['done'])

                        self.agent.multi_update(self.agent_updates_per_step,
                                                self.virtual_replay_buffer,
                                                self.logger)

                        actions_this_step[Actions.GENERATE_ROLLOUTS] = 1

                        if take_random_action:
                            actions_this_step[Actions.RANDOM_ACTION] = 1

                    else:
                        self.agent.multi_update(self.agent_updates_per_step,
                                                self.real_replay_buffer,
                                                self.logger)

                    if self.agent_updates_per_step > 0:
                        agent_update_performed = True
                        actions_this_step[Actions.UPDATE_AGENT] = 1

                action_log.append(actions_this_step)
                step_total += 1

            print('')

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                self.logger.save_state({'env': self.env}, None)

            test_return = 0

            if step_total > self.init_steps or epoch < 1:
                # Test the performance of the deterministic version of the agent.
                test_return = test_agent(self.test_env,
                                         self.agent,
                                         self.max_ep_len,
                                         self.num_test_episodes,
                                         self.logger,
                                         self.render and
                                         step_total > self.init_steps)

                tested_agent = True

            test_performances.append([epoch, test_return])

            log_end_of_epoch(self.logger, epoch, step_total, start_time,
                             agent_update_performed, model_trained_this_epoch,
                             rollout_length, episode_finished, tested_agent)

        return torch.as_tensor(test_performances, dtype=torch.float32), \
            torch.as_tensor(action_log, dtype=torch.float32)


def log_end_of_epoch(logger, epoch, step_total, start_time,
                     agent_update_performed, model_trained, rollout_length,
                     episode_finished, tested_agent):

    logger.log_tabular('Epoch', epoch, epoch)

    logger.log_tabular('RolloutLength', epoch, rollout_length)

    if not episode_finished:
        logger.store(EpRet=0)
        logger.store(EpLen=0)

    logger.log_tabular('EpRet', epoch, with_min_and_max=True)
    logger.log_tabular('EpLen', epoch, average_only=True)

    if not tested_agent:
        logger.store(TestEpRet=0)
        logger.store(TestEpLen=0)

    logger.log_tabular('TestEpRet', epoch, with_min_and_max=True)
    logger.log_tabular('TestEpLen', epoch, average_only=True)

    logger.log_tabular('TotalEnvInteracts', epoch, step_total)

    # Use placeholder value if no model update has been performed yet
    if not model_trained:
        logger.store(LossEnvModel=0)

    logger.log_tabular('LossEnvModel', epoch, with_min_and_max=True)

    # Use placeholder values if no agent update has been performed yet
    if not agent_update_performed:
        logger.store(Q1Vals=0,
                     Q2Vals=0,
                     LogPi=0,
                     LossPi=0,
                     LossQ=0)

    logger.log_tabular('Q1Vals', epoch, with_min_and_max=True)
    logger.log_tabular('Q2Vals', epoch, with_min_and_max=True)
    logger.log_tabular('LogPi', epoch, with_min_and_max=True)
    logger.log_tabular('LossPi', epoch, average_only=True)
    logger.log_tabular('LossQ', epoch, average_only=True)

    logger.log_tabular('Time', epoch, time.time()-start_time)
    logger.dump_tabular()

    logger.save_replay_buffer_to_tensorboard(epoch, mode='umaze')
