import time
from enum import IntEnum
from typing import Optional, Union, cast

import gym
import numpy as np
import torch

from offline_mbrl.actors.behavioral_cloning import (
    BehavioralCloningAgent,
    BehavioralCloningConfiguration,
)
from offline_mbrl.actors.sac import SAC
from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.schemas import (
    AgentConfiguration,
    EnvironmentModelConfiguration,
    EpochLoggerConfiguration,
    SACConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.utils.evaluate_agent_performance import evaluate_agent_performance
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.logx import EpochLogger
from offline_mbrl.utils.model_needs_training import model_needs_training
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.termination_functions import get_termination_function
from offline_mbrl.utils.virtual_rollouts import generate_virtual_rollouts


class Trainer:
    def __init__(
        self,
        config: TrainerConfiguration = TrainerConfiguration(),
        agent_config: AgentConfiguration = SACConfiguration(),
        env_model_config: EnvironmentModelConfiguration = EnvironmentModelConfiguration(),  # pylint: disable=line-too-long
        logger_config: EpochLoggerConfiguration = EpochLoggerConfiguration(),
    ):
        self.config = config
        self.env_model_config = env_model_config
        self.agent_config = agent_config
        self.logger_config = logger_config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        logger_config.env_name = config.env_name
        self.logger = EpochLogger(logger_config)
        local_vars = locals()
        self.logger.save_config(
            {key: local_vars[key] for key in local_vars if key != "self"}
        )

        self.env_name = config.env_name
        self.env = gym.make(config.env_name)
        self.env.reset(seed=config.seed)
        self.env.action_space.seed(config.seed)

        self.real_replay_buffer, self.virtual_replay_buffer = self._initialize_buffers()
        self.env_model = self._initialize_env_model()
        self.agent = self._initialize_agent()

        self.logger.setup_pytorch_saver({"agent": self.agent})

        self.total_steps = config.steps_per_epoch * (
            config.online_epochs + config.offline_epochs
        )
        self.max_episode_length = min(config.max_episode_length, self.total_steps)

        self.steps_since_model_training = int(1e10)
        self.model_trained_last_epoch = False
        self.actions_last_step = [0 for i in range(len(Actions))]

    def _initialize_buffers(self) -> tuple[ReplayBuffer, ReplayBuffer]:
        real_replay_buffer, obs_dim, act_dim = load_dataset_from_env(
            self.env_name,
            n_samples=self.config.n_samples_from_dataset,
            buffer_size=self.config.real_buffer_size,
            buffer_device=self.config.device,
        )
        virtual_replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=self.config.virtual_buffer_size,
            device=self.config.device,
        )

        return real_replay_buffer, virtual_replay_buffer

    def _initialize_agent(self) -> Union[BehavioralCloningAgent, SAC]:
        if self.config.pretrained_agent_path is not None:
            agent = torch.load(
                self.config.pretrained_agent_path, map_location=self.config.device
            )
        else:
            self.agent_config.preprocessing_function = get_preprocessing_function(
                self.config.env_name, self.config.device
            )

            if self.agent_config.type == "bc":
                assert isinstance(self.agent_config, BehavioralCloningConfiguration)
                agent = BehavioralCloningAgent(
                    self.env.observation_space, self.env.action_space, self.agent_config
                )
            elif self.agent_config.type == "sac":
                assert isinstance(self.agent_config, SACConfiguration)
                agent = SAC(
                    self.env.observation_space,
                    self.env.action_space,
                    self.agent_config,
                )
            else:
                raise ValueError(f"Unknown agent type: {self.agent_config.type}")

        return agent

    def _initialize_env_model(self) -> Optional[EnvironmentModel]:
        env_model: Optional[EnvironmentModel] = None

        if self.config.use_env_model:
            if self.config.pretrained_env_model_path is not None:
                env_model = cast(
                    EnvironmentModel,
                    torch.load(
                        self.config.pretrained_env_model_path,
                        map_location=self.config.device,
                    ),
                )

            else:
                self.env_model_config.preprocessing_function = (
                    get_preprocessing_function(self.config.env_name, self.config.device)
                )
                self.env_model_config.termination_function = get_termination_function(
                    self.config.env_name
                )

                assert self.env.observation_space.shape is not None
                assert self.env.action_space.shape is not None

                env_model = EnvironmentModel(
                    self.env.observation_space.shape[0],
                    self.env.action_space.shape[0],
                    self.env_model_config,
                )

        return env_model

    def train(
        self, tuning: bool = False, quiet: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:

        start_time = time.time()
        obs = torch.as_tensor(self.env.reset())
        episode_return = 0.0
        episode_length = 0

        if self.total_steps < self.config.init_steps:
            raise ValueError(
                """Number of total steps lower than init steps.
                Increase number of epochs or steps per epoch."""
            )

        step_total = -self.config.offline_epochs * self.config.steps_per_epoch

        test_performances = []
        action_log = []

        next_obs = None

        for epoch in range(
            -self.config.offline_epochs + 1, self.config.online_epochs + 1
        ):

            self.model_trained_last_epoch = False
            agent_update_performed = False
            at_least_one_real_episode_finished = False
            tested_agent = False

            if not quiet:
                print(f"Epoch {epoch}")

            for step_this_epoch in range(self.config.steps_per_epoch):
                self.actions_last_step = [0 for i in range(len(Actions))]

                take_random_action = (
                    step_total
                    + self.config.offline_epochs * self.config.steps_per_epoch
                    < self.config.random_steps
                )

                self.train_model_if_necessary(step_total, self.env_model)

                if (step_this_epoch + 1) % 10 == 0 and not tuning and not quiet:
                    print(
                        f"Epoch {epoch}, step "
                        f"{step_this_epoch+1}/{self.config.steps_per_epoch}",
                        end="\r",
                    )

                if epoch > 0:
                    (
                        obs,
                        episode_return,
                        episode_length,
                        episode_finished,
                    ) = self._interact_with_real_environment(
                        obs, episode_return, episode_length, take_random_action
                    )

                    at_least_one_real_episode_finished = (
                        episode_finished or at_least_one_real_episode_finished
                    )

                if (
                    step_total >= self.config.init_steps
                    or self.config.offline_epochs > 0
                ):
                    next_obs = (
                        self._generate_virtual_rollouts_if_necessary_and_update_agent(
                            next_obs, take_random_action
                        )
                    )
                    agent_update_performed = True
                    self.actions_last_step[Actions.UPDATE_AGENT] = 1

                action_log.append(self.actions_last_step)
                step_total += 1

            print("")

            # Save model
            if (epoch % self.config.save_frequency == 0) or (
                epoch == self.config.online_epochs
            ):
                self.logger.save_state({"env": self.env}, None)

            test_return = 0.0

            if step_total > self.config.init_steps or self.config.offline_epochs > 0:

                # Test the performance of the deterministic version of the agent.
                test_return = evaluate_agent_performance(
                    self.env,
                    self.agent,
                    self.config.test_episodes,
                    self.max_episode_length,
                    self.logger,
                    self.config.render_test_episodes
                    and step_total > self.config.init_steps,
                )

                tested_agent = True

            test_performances.append([epoch, test_return])

            log_end_of_epoch(
                self.logger,
                epoch,
                step_total,
                start_time,
                agent_update_performed,
                self.model_trained_last_epoch,
                at_least_one_real_episode_finished,
                tested_agent,
            )

        return torch.as_tensor(test_performances, dtype=torch.float32), torch.as_tensor(
            action_log, dtype=torch.float32
        )

    def _interact_with_real_environment(
        self,
        obs: torch.Tensor,
        episode_return: float,
        episode_length: int,
        take_random_action: bool,
    ) -> tuple[torch.Tensor, float, int, bool]:
        episode_finished = False

        for _ in range(self.config.env_interactions_per_step):
            if take_random_action:
                action = self.env.action_space.sample()
                self.actions_last_step[Actions.RANDOM_ACTION] = 1
            else:
                action = self.agent.act(obs).cpu().numpy()

            next_obs, reward, done, _ = self.env.step(action)
            episode_return += reward
            episode_length += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if episode_length == self.max_episode_length else done

            self.real_replay_buffer.store(
                torch.as_tensor(obs),
                torch.as_tensor(action),
                torch.as_tensor(reward),
                torch.as_tensor(next_obs),
                torch.as_tensor(done),
            )
            obs = next_obs

            # End of trajectory handling
            if done or (episode_length == self.max_episode_length):
                episode_finished = True
                self.logger.store(EpRet=episode_return, EpLen=episode_length)
                obs = torch.as_tensor(self.env.reset())
                episode_return = 0
                episode_length = 0

        self.steps_since_model_training += 1
        self.actions_last_step[Actions.INTERACT_WITH_ENV] = 1

        return obs, episode_return, episode_length, episode_finished

    def _generate_virtual_rollouts_if_necessary_and_update_agent(
        self, prev_obs: Optional[dict], take_random_action: bool
    ) -> Optional[dict]:
        if self.env_model is not None and self.config.n_parallel_virtual_rollouts > 0:
            for _ in range(self.config.agent_updates_per_step):
                rollouts, prev_obs = generate_virtual_rollouts(
                    self.env_model,
                    self.agent,
                    self.real_replay_buffer,
                    steps=1,
                    n_rollouts=self.config.n_parallel_virtual_rollouts,
                    pessimism=self.env_model_config.pessimism,
                    ood_threshold=self.env_model_config.ood_threshold,
                    mode=self.env_model_config.mode,
                    random_action=take_random_action,
                    prev_obs=prev_obs,
                    max_rollout_length=self.config.max_virtual_rollout_length,
                )

                self.virtual_replay_buffer.store_batch(
                    rollouts["obs"],
                    rollouts["act"],
                    rollouts["rew"],
                    rollouts["next_obs"],
                    rollouts["done"],
                )

                self.agent.multi_update(1, self.virtual_replay_buffer, self.logger)

            self.actions_last_step[Actions.GENERATE_ROLLOUTS] = 1

            if take_random_action:
                self.actions_last_step[Actions.RANDOM_ACTION] = 1

        else:
            self.agent.multi_update(
                self.config.agent_updates_per_step,
                self.real_replay_buffer,
                self.logger,
            )

        return prev_obs

    def train_model_if_necessary(
        self, step: int, model: Optional[EnvironmentModel]
    ) -> None:
        # Train environment model on real experience
        if model_needs_training(
            model,
            step,
            self.real_replay_buffer.size,
            self.config.init_steps,
            self.steps_since_model_training,
            self.config.train_env_model_every,
        ):

            if self.config.train_env_model_from_scratch:
                assert isinstance(self.env_model, EnvironmentModel)
                self.env_model = EnvironmentModel(
                    self.env_model.obs_dim,
                    self.env_model.act_dim,
                    self.env_model_config,
                )

            assert self.env_model is not None

            model_val_error, _ = self.env_model.train_to_convergence(
                self.real_replay_buffer, self.env_model_config
            )

            if self.config.reset_virtual_buffer_after_env_model_training:
                self.virtual_replay_buffer.clear()

            model_val_error = model_val_error.mean()
            self.model_trained_last_epoch = True
            self.steps_since_model_training = 0
            self.logger.store(LossEnvModel=model_val_error.item())
            self.actions_last_step[Actions.TRAIN_MODEL] = 1
            print("")


def log_end_of_epoch(
    logger: EpochLogger,
    epoch: int,
    step_total: int,
    start_time: float,
    agent_update_performed: bool,
    model_trained: bool,
    episode_finished: bool,
    tested_agent: bool,
) -> None:

    logger.log_tabular("Epoch", epoch, epoch)

    # Use placeholder value if no interaction with the real environment has happened yet
    if not episode_finished:
        logger.store(EpRet=0)
        logger.store(EpLen=0)

    logger.log_tabular("EpRet", epoch, with_min_and_max=True)
    logger.log_tabular("EpLen", epoch, average_only=True)

    # Use placeholder value if the agent has not been tested yet
    if not tested_agent:
        logger.store(TestEpRet=0)
        logger.store(TestEpLen=0)

    logger.log_tabular("TestEpRet", epoch, with_min_and_max=True)
    logger.log_tabular("TestEpLen", epoch, average_only=True)

    logger.log_tabular("TotalEnvInteracts", epoch, step_total)

    # Use placeholder value if no model update has been performed yet
    if not model_trained:
        logger.store(LossEnvModel=0)

    logger.log_tabular("LossEnvModel", epoch, with_min_and_max=True)

    # Use placeholder values if no agent update has been performed yet
    if not agent_update_performed:
        logger.store(Q1Vals=0, Q2Vals=0, LogPi=0, LossPi=0, LossQ=0)

    logger.log_tabular("Q1Vals", epoch, with_min_and_max=True)
    logger.log_tabular("Q2Vals", epoch, with_min_and_max=True)
    logger.log_tabular("LogPi", epoch, with_min_and_max=True)
    logger.log_tabular("LossPi", epoch, average_only=True)
    logger.log_tabular("LossQ", epoch, average_only=True)

    logger.log_tabular("Time", epoch, time.time() - start_time)
    logger.dump_tabular()


class Actions(IntEnum):
    """Actions that can be performed during an iteration of the training loop.

    Used for logging what happened during training and testing the training loop.
    """

    TRAIN_MODEL = 0
    UPDATE_AGENT = 1
    RANDOM_ACTION = 2
    GENERATE_ROLLOUTS = 3
    INTERACT_WITH_ENV = 4
