import argparse
from typing import Optional

import ray
from ax.service.ax_client import AxClient
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.search.ax import AxSearch

from offline_mbrl.schemas import (
    AgentConfiguration,
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


def training_function(
    trainer_config: TrainerConfiguration,
    agent_config: AgentConfiguration,
    env_model_config: EnvironmentModelConfiguration,
    logger_config: EpochLoggerConfiguration,
    tuning=True,
):
    trainer = Trainer(trainer_config, agent_config, env_model_config, logger_config)
    return trainer.train(tuning=tuning, quiet=True)


@ray.remote(num_gpus=0.5, max_retries=3)
def training_wrapper(config, seed):
    exp_name = get_experiment_name(config)

    config.update(
        seed=seed,
        logger_kwargs=setup_logger_kwargs(exp_name, seed=seed),
    )
    return training_function(config, tuning=False)


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

    if not args.tune:
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
        training_function(
            trainer_config, agent_config, env_model_config, logger_config, tuning=False
        )

    else:
        # TODO: Remove tuning
        config.update(
            epochs=30,
            steps_per_epoch=15000,
            agent_kwargs=dict(
                type=agent_type,
                batch_size=256,
                agent_hidden=128,
                gamma=0.99,
                pi_lr=3e-4,
                q_lr=3e-4,
            ),
        )

        parameters = [
            {
                "name": "max_rollout_length",
                "type": "range",
                "bounds": [1, 20],
                "value_type": "int",
                "log_scale": False,
            }
        ]

        (
            r_max,
            max_uncertainty,
            mean_uncertainty,
            std_uncertainty,
        ) = get_uncertainty_distribution(args.env_name, args.mode)

        r_max = float(r_max)
        max_uncertainty = float(max_uncertainty)
        mean_uncertainty = float(mean_uncertainty)
        std_uncertainty = float(std_uncertainty)

        print(
            f"R_max: {r_max}, Max uncertainty: {max_uncertainty}, "
            f"Mean uncertainty: {mean_uncertainty}"
        )

        if args.mode in PARTITIONING_MODES:
            parameters += [
                {
                    "name": "ood_threshold",
                    "type": "range",
                    "bounds": [mean_uncertainty, max_uncertainty],
                    "value_type": "float",
                    "log_scale": True,
                }
            ]

            config.update(model_pessimism=0)

        elif args.mode in PENALTY_MODES:
            parameters += [
                {
                    "name": "model_pessimism",
                    "type": "range",
                    "bounds": [0, r_max / max_uncertainty],
                    "value_type": "float",
                    "log_scale": False,
                }
            ]

            config.update(ood_threshold=0)

        assert config["agent_kwargs"]["batch_size"] is not None
        assert config["agent_kwargs"]["agent_hidden"] is not None
        assert config["agent_kwargs"]["gamma"] is not None
        assert config["agent_kwargs"]["pi_lr"] is not None
        assert config["agent_kwargs"]["q_lr"] is not None

        ray.init()
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="avg_test_return",
            mode="max",
            max_t=10,
            grace_period=5,
            reduction_factor=3,
            brackets=1,
        )

        search_alg = AxSearch(
            ax_client=AxClient(enforce_sequential_optimization=False),
            space=parameters,
            metric="avg_test_return",
            mode="max",
        )

        analysis = tune.run(
            tune.with_parameters(training_function),
            name=args.env_name
            + "-"
            + config["mode"]
            + "-tuning-lvl-"
            + str(args.level),
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.n_trials,
            config=config,
            max_failures=3,
            resources_per_trial={"gpu": 0.5},
        )

        print(
            "Best config: ",
            analysis.get_best_config(metric="avg_test_return", mode="max"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default=SAC)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tune", type=str2bool, default=False)

    main(parser.parse_args())
