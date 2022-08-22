import argparse
import os

import ray
import torch
from ax.service.ax_client import AxClient
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.ax import AxSearch

from offline_mbrl.train import Trainer
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.envs import HYPERPARAMS
from offline_mbrl.utils.modes import (
    BEHAVIORAL_CLONING,
    MBPO,
    MODES,
    PARTITIONING_MODES,
    PENALTY_MODES,
    SAC,
)
from offline_mbrl.utils.postprocessing import get_postprocessing_function
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.setup_logger_kwargs import setup_logger_kwargs
from offline_mbrl.utils.str2bool import str2bool
from offline_mbrl.utils.uncertainty_distribution import get_uncertainty_distribution


def training_function(config, tuning=True):
    config["agent_kwargs"].update(
        {"hidden": 4 * [config["agent_kwargs"]["agent_hidden"]]}
    )
    trainer = Trainer(**config)
    return trainer.train(tuning=tuning, silent=True)


def get_exp_name(config):
    exp_name = config.env_name + "-" + config["mode"]

    if (config["mode"] == MBPO or config["mode"] == SAC) and config[
        "env_steps_per_step"
    ] == 0:
        exp_name += "-offline"

    if config["mode"] not in [BEHAVIORAL_CLONING, SAC]:
        exp_name += (
            "-"
            + str(config["rollouts_per_step"])
            + "rollouts"
            + "-"
            + str(config["max_rollout_length"])
            + "steps"
        )

        if config["mode"] in PENALTY_MODES:
            exp_name += "-" + str(config["model_pessimism"]) + "pessimism"

        if config["mode"] in PARTITIONING_MODES:
            exp_name += "-" + str(config["ood_threshold"]) + "threshold"

    if config["pretrained_interaction_agent_path"] != "":
        exp_name += "-double_agent-" + str(config["exploration_chance"])

    if config["pretrained_agent_path"] != "":
        exp_name += "-pretrained"

    exp_name += "-" + str(config["n_samples_from_dataset"]) + "samples"

    return exp_name


@ray.remote(num_gpus=0.5, max_retries=3)
def training_wrapper(config, seed):
    print("hi")
    exp_name = get_exp_name(config)

    config.update(
        seed=seed,
        logger_kwargs=setup_logger_kwargs(exp_name, seed=seed),
    )
    return training_function(config, tuning=False)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device != "":
        device = args.device

    pretrained_model_name = args.env_name + "-model.pt"

    if args.mode not in MODES:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.mode == BEHAVIORAL_CLONING:
        use_model = False
        agent_type = "bc"
    elif args.mode == SAC:
        use_model = False
        agent_type = "sac"
    elif args.mode == MBPO:
        use_model = True
        agent_type = "sac"
    else:
        use_model = True
        agent_type = "sac"

    # None values must be filled for tuning and final training
    config = dict(
        env_name=args.env_name,
        agent_kwargs=dict(
            type=agent_type,
            batch_size=None,
            agent_hidden=None,
            gamma=None,
            pi_lr=None,
            q_lr=None,
        ),
        max_rollout_length=None,
        model_pessimism=None,
        ood_threshold=None,
        rollouts_per_step=args.n_rollouts,
        model_kwargs=dict(
            in_normalized_space=True,
            lr=1e-3,
            batch_size=256,
            hidden=[200, 200, 200, 200],
            type="probabilistic",
            n_networks=7,
            pre_fn=get_preprocessing_function(args.env_name),
            post_fn=get_postprocessing_function(args.env_name),
            no_reward=False,
            use_batch_norm=False,
        ),
        dataset_path="",
        seed=0,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        random_steps=args.random_steps,
        init_steps=args.init_steps,
        env_steps_per_step=args.env_steps_per_step,
        n_samples_from_dataset=args.n_samples_from_dataset,
        agent_updates_per_step=args.agent_updates_per_step,
        num_test_episodes=args.num_test_episodes,
        curriculum=[1, 1, 20, 100],
        max_ep_len=1000,
        use_model=use_model,
        pretrained_agent_path="",
        pretrained_model_path=os.path.join(MODELS_DIR, pretrained_model_name)
        if not args.new_model
        else "",
        mode=args.mode,
        model_max_n_train_batches=args.max_n_train_batches,
        rollout_schedule=[1, 1, 20, 100],
        continuous_rollouts=True,
        train_model_every=args.train_model_every,
        use_custom_reward=False,
        real_buffer_size=int(2e6),
        virtual_buffer_size=args.virtual_buffer_size,
        reset_buffer=args.reset_buffer,
        virtual_pretrain_epochs=0,
        train_model_from_scratch=args.train_model_from_scratch,
        reset_maze2d_umaze=False,
        pretrain_epochs=0,
        setup_test_env=False,
        logger_kwargs={},
        save_freq=1,
        device=device,
        render=False,
    )

    if args.level == 0:

        if args.tuned_params:
            if args.mode in PARTITIONING_MODES:
                (rollouts_per_step, max_rollout_length, ood_threshold) = HYPERPARAMS[
                    args.mode
                ][args.env_name]
                model_pessimism = 0
            elif args.mode in PENALTY_MODES:
                (rollouts_per_step, max_rollout_length, model_pessimism) = HYPERPARAMS[
                    args.mode
                ][args.env_name]
                ood_threshold = 0
        else:
            rollouts_per_step = args.n_rollouts
            max_rollout_length = args.rollout_length
            model_pessimism = args.pessimism
            ood_threshold = args.ood_threshold

        # Basic config
        config.update(
            agent_kwargs=dict(
                type=agent_type,
                batch_size=256,
                agent_hidden=args.n_hidden,
                gamma=0.99,
                pi_lr=3e-4,
                q_lr=3e-4,
            ),
            rollouts_per_step=rollouts_per_step,
            max_rollout_length=max_rollout_length,
            model_pessimism=model_pessimism,
            ood_threshold=ood_threshold,
            pretrained_agent_path=args.pretrained_agent_path,
            pretrained_interaction_agent_path=args.pretrained_interaction_agent_path,
            interaction_agent_pessimism=args.interaction_pessimism,
            interaction_agent_threshold=args.interaction_threshold,
            exploration_chance=args.exploration_chance,
        )

        assert config["agent_kwargs"]["batch_size"] is not None
        assert config["agent_kwargs"]["agent_hidden"] is not None
        assert config["agent_kwargs"]["gamma"] is not None
        assert config["agent_kwargs"]["pi_lr"] is not None
        assert config["agent_kwargs"]["q_lr"] is not None
        assert config["rollouts_per_step"] is not None
        assert config["max_rollout_length"] is not None
        assert config["model_pessimism"] is not None
        assert config["ood_threshold"] is not None

        if args.use_ray:
            ray.init()
            unfinished_jobs = []

            for seed in range(args.start_seed, args.start_seed + args.seeds):
                job_id = training_wrapper.remote(config, seed)
                unfinished_jobs.append(job_id)

            while unfinished_jobs:
                _, unfinished_jobs = ray.wait(unfinished_jobs)
        else:
            for seed in range(args.start_seed, args.start_seed + args.seeds):
                config.update(
                    seed=seed,
                    logger_kwargs=setup_logger_kwargs(get_exp_name(config), seed=seed),
                )
                training_function(config, tuning=False)

    else:
        if args.level == 1:
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
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--tuned_params", type=str2bool, default=False)
    parser.add_argument("--new_model", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default=SAC)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_test_episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--train_model_every", type=int, default=0)
    parser.add_argument("--train_model_from_scratch", type=str2bool, default=False)
    parser.add_argument("--reset_buffer", type=str2bool, default=False)
    parser.add_argument("--max_n_train_batches", type=int, default=-1)
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--env_steps_per_step", type=int, default=0)
    parser.add_argument("--random_steps", type=int, default=8000)
    parser.add_argument("--init_steps", type=int, default=4000)
    parser.add_argument("--pessimism", type=float, default=1)
    parser.add_argument("--ood_threshold", type=float, default=0.5)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--rollout_length", type=int, default=15)
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_samples_from_dataset", type=int, default=-1)
    parser.add_argument("--agent_updates_per_step", type=int, default=1)
    parser.add_argument("--pretrained_agent_path", type=str, default="")
    parser.add_argument("--pretrained_interaction_agent_path", type=str, default="")
    parser.add_argument("--interaction_pessimism", type=float, default=1)
    parser.add_argument("--interaction_threshold", type=float, default=0.5)
    parser.add_argument("--exploration_chance", type=float, default=1)
    parser.add_argument("--virtual_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--use_ray", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="")

    main(parser.parse_args())
