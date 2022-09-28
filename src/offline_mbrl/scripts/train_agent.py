import argparse
from typing import Optional

from offline_mbrl.schemas import (
    BehavioralCloningConfiguration,
    EnvironmentModelConfiguration,
    EpochLoggerConfiguration,
    SACConfiguration,
    TrainerConfiguration,
)
from offline_mbrl.train import Trainer
from offline_mbrl.user_config import DATA_DIR, MODELS_DIR
from offline_mbrl.utils.envs import HOPPER_MEDIUM_REPLAY_V2
from offline_mbrl.utils.get_experiment_name import get_experiment_name
from offline_mbrl.utils.hyperparameters import HYPERPARAMS
from offline_mbrl.utils.modes import (
    ALL_MODES,
    BEHAVIORAL_CLONING,
    MBPO,
    PARTITIONING_MODES,
    PENALTY_MODES,
    SAC,
)
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.setup_logger_kwargs import setup_logger_kwargs
from offline_mbrl.utils.str2bool import str2bool
from offline_mbrl.utils.termination_functions import get_termination_function
from offline_mbrl.utils.uncertainty_distribution import get_uncertainty_distribution


def main(args: argparse.Namespace):

    trainer_config = TrainerConfiguration()

    trainer_config.env_name = args.env_name

    if args.mode not in ALL_MODES:
        raise ValueError(f"Unknown mode: {args.mode}")

    preprocessing_function = get_preprocessing_function(trainer_config.env_name)
    termination_function = get_termination_function(trainer_config.env_name)

    env_model_config: Optional[
        EnvironmentModelConfiguration
    ] = EnvironmentModelConfiguration(
        type="probabilistic",
        n_networks=8,
        preprocessing_function=preprocessing_function,
        termination_function=termination_function,
    )

    agent_config = SACConfiguration(preprocessing_function=preprocessing_function)

    if args.mode == BEHAVIORAL_CLONING:
        trainer_config.use_env_model = False
        agent_config = BehavioralCloningConfiguration(
            preprocessing_function=preprocessing_function
        )
    elif args.mode == SAC:
        trainer_config.use_env_model = False
    elif args.mode == MBPO:
        trainer_config.use_env_model = True
    else:
        trainer_config.use_env_model = True

    if args.mode in PARTITIONING_MODES:
        (
            trainer_config.max_virtual_rollout_length,
            env_model_config.ood_threshold,
        ) = HYPERPARAMS[args.mode][args.env_name]
        env_model_config.pessimism = 0
    elif args.mode in PENALTY_MODES:
        (
            trainer_config.max_virtual_rollout_length,
            env_model_config.pessimism,
        ) = HYPERPARAMS[args.mode][args.env_name]
        env_model_config.ood_threshold = 0

    logger_config = EpochLoggerConfiguration(
        **setup_logger_kwargs(
            exp_name=get_experiment_name(trainer_config, args.mode),
            seed=trainer_config.seed,
            data_dir=DATA_DIR,
        )
    )

    trainer = Trainer(trainer_config, agent_config, env_model_config, logger_config)
    return trainer.train(quiet=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default=SAC)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    main(parser.parse_args())
