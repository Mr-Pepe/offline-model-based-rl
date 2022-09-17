from pathlib import Path
from typing import Callable, Literal, Optional, Type

from pydantic import BaseModel, PositiveInt
from torch import nn

from offline_mbrl.utils.envs import HOPPER_RANDOM_V2


class TrainerConfiguration(BaseModel):
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

        max_episode_length (int): Maximum length of trajectory / episode / rollout.

        use_model (bool): Whether to augment data with virtual rollouts.

        n_parallel_rollouts (int): The number of model rollouts to perform per
            environment step.

        train_model_every (int): After how many steps the model should be
            retrained.

        agent_updates_per_step (int): The number of agent updates to
            perform per environment step.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    env_name: str = HOPPER_RANDOM_V2
    seed: int = 0
    online_epochs: int = 100
    offline_epochs: int = 0
    steps_per_epoch: int = 4_000
    random_steps: int = 10_000
    init_steps: int = 1_000
    env_interactions_per_step: int = 1
    agent_updates_per_step: PositiveInt = 1
    test_episodes: int = 10
    render_test_episodes: bool = False
    max_episode_length: int = 1000
    use_env_model: bool = False
    n_parallel_virtual_rollouts: int = 10
    max_virtual_rollout_length: Optional[int] = None
    continuous_rollouts: bool = True
    train_env_model_every: int = 250
    train_env_model_from_scratch: bool = False
    env_model_checkpoint_dir: Optional[Path] = None
    reset_virtual_buffer_after_env_model_training: bool = False
    pretrained_env_model_path: Optional[Path] = None
    pretrained_agent_path: Optional[Path] = None
    n_samples_from_dataset: Optional[int] = 0
    real_buffer_size: int = int(1e6)
    virtual_buffer_size: int = int(1e6)
    save_frequency: int = 1
    device: str = "cpu"


class EnvironmentModelConfiguration(BaseModel):
    type: Literal["deterministic", "probabilistic"] = "deterministic"
    n_networks: int = 1
    hidden_layer_sizes: tuple[int, ...] = (128, 128)
    pessimism: float = 0
    ood_threshold: float = -1
    mode: Optional[str] = None
    training_batch_size: int = 256
    training_patience: int = 1
    lr: float = 1e-3
    val_split: float = 0.2
    max_number_of_training_batches: Optional[int] = None
    max_number_of_training_epochs: Optional[int] = None
    preprocessing_function: Optional[Callable] = None
    termination_function: Optional[Callable] = None
    reward_function: Optional[Callable] = None
    obs_bounds_trainable: bool = True
    reward_bounds_trainable: bool = True
    device: str = "cpu"


class AgentConfiguration(BaseModel):
    type: Literal["sac", "bc"] = "sac"
    preprocessing_function: Optional[Callable] = None
    hidden_layer_sizes: tuple[int, ...] = (256, 256)
    activation: Type[nn.Module] = nn.ReLU
    training_batch_size: int = 256
    device: str = "cpu"


class BehavioralCloningConfiguration(AgentConfiguration):
    type: Literal["bc"] = "bc"
    lr: float = 3e-4


class SACConfiguration(AgentConfiguration):
    type: Literal["sac"] = "sac"
    pi_lr: float = 3e-4
    q_lr: float = 3e-4
    gamma: float = 0.99
    alpha: float = 0.2
    polyak: float = 0.995


class EpochLoggerConfiguration(BaseModel):
    """Configures an epoch logger.

    Args:
        output_dir (Optional[Path]): A directory for saving results to. If ``None``,
            defaults to a temp directory of the form
            ``/tmp/experiments/somerandomnumber``. Defaults to None.
        output_filename (str): Name for the tab-separated-value file
            containing metrics logged throughout a training run.
            Defaults to ``progress.txt``.
        experiment_name (Optional[str]): Experiment name. If you run multiple training
            runs and give them all the same ``experiment_name``, the plotter
            will know to group them. (Use case: if you run the same
            hyperparameter configuration with multiple random seeds, you
            should give them all the same ``experiment_name``). Defaults to None.
        env_Name (str): The environment name used in the experiment. Defaults to an
            empty string.
    """

    output_dir: Optional[Path] = None
    output_filename: str = "progress.txt"
    experiment_name: Optional[str] = None
    env_name: str = ""
