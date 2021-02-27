import argparse
from benchmark.utils.uncertainty_distribution import get_uncertainty_distribution
from benchmark.utils.modes import ALEATORIC_PENALTY, PARTITIONING_MODES, PENALTY_MODES, MODES, UNDERESTIMATION
from benchmark.utils.run_utils import setup_logger_kwargs
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from benchmark.utils.envs import \
    HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_REPLAY, HOPPER_MEDIUM, \
    HOPPER_MEDIUM_EXPERT, HYPERPARAMS, WALKER_ENVS, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_REPLAY, WALKER_MEDIUM_REPLAY_V2
from benchmark.user_config import MODELS_DIR
from benchmark.train import Trainer
from benchmark.utils.str2bool import str2bool
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
import torch
import os


def training_function(config, tuning=True):
    config["sac_kwargs"].update(
        {"hidden": 4*[config["sac_kwargs"]["agent_hidden"]]})
    trainer = Trainer(**config)
    return trainer.train(tuning=tuning, silent=True)


@ray.remote(num_gpus=0.5, max_retries=3)
def training_wrapper(config, seed):
    print("hi")
    exp_name = args.env_name+'-' + \
        config['mode'] + '-' + str(config['rollouts_per_step']) + \
        'rollouts' + '-' + str(config['max_rollout_length']) + 'steps'

    if config['mode'] in PENALTY_MODES:
        exp_name += '-' + str(config['model_pessimism']) + 'pessimism'

    if config['mode'] in PARTITIONING_MODES:
        exp_name += '-' + str(config['ood_threshold']) + 'threshold'

    config.update(
        epochs=args.epochs,
        seed=seed,
        logger_kwargs=setup_logger_kwargs(exp_name,
                                          seed=seed),
    )
    return training_function(config, tuning=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default=WALKER_MEDIUM_REPLAY_V2)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--tuned_params', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default=UNDERESTIMATION)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--pessimism', type=float, default=1)
    parser.add_argument('--ood_threshold', type=float, default=0.5)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--rollout_length', type=int, default=100)
    parser.add_argument('--n_rollouts', type=int, default=50)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--pretrained_agent_path', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device != '':
        device = args.device

    pretrained_model_name = args.env_name + '-model.pt'

    if args.mode not in MODES:
        raise ValueError("Unknown mode: {}".format(args.mode))

    # None values must be filled for tuning and final training
    config = dict(
        env_name=args.env_name,
        sac_kwargs=dict(batch_size=None,
                        agent_hidden=None,
                        gamma=None,
                        pi_lr=None,
                        q_lr=None,
                        ),
        max_rollout_length=None,
        model_pessimism=None,
        ood_threshold=None,
        rollouts_per_step=args.n_rollouts,
        model_kwargs=dict(),
        dataset_path='',
        seed=0,
        epochs=args.epochs,
        steps_per_epoch=15000,
        random_steps=8000,
        init_steps=4000,
        env_steps_per_step=0,
        n_samples_from_dataset=-1,
        agent_updates_per_step=1,
        num_test_episodes=20,
        curriculum=[1, 1, 20, 100],
        max_ep_len=1000,
        use_model=True,
        pretrained_agent_path='',
        pretrained_model_path=os.path.join(
            MODELS_DIR, pretrained_model_name),
        mode=args.mode,
        model_max_n_train_batches=-1,
        rollout_schedule=[1, 1, 20, 100],
        continuous_rollouts=True,
        train_model_every=0,
        use_custom_reward=False,
        real_buffer_size=int(2e6),
        virtual_buffer_size=int(1e6),
        reset_buffer=False,
        virtual_pretrain_epochs=0,
        train_model_from_scratch=False,
        reset_maze2d_umaze=False,
        pretrain_epochs=0,
        setup_test_env=False,
        logger_kwargs=dict(),
        save_freq=1,
        device=device,
        render=False)

    if args.level == 0:

        if args.tuned_params:
            if args.mode in PARTITIONING_MODES:
                (rollouts_per_step, max_rollout_length,
                 ood_threshold) = HYPERPARAMS[args.mode][args.env_name]
                model_pessimism = 0
            elif args.mode in PENALTY_MODES:
                (rollouts_per_step, max_rollout_length,
                 model_pessimism) = HYPERPARAMS[args.mode][args.env_name]
                ood_threshold = 0
        else:
            rollouts_per_step = args.n_rollouts
            max_rollout_length = args.rollout_length
            model_pessimism = args.pessimism
            ood_threshold = args.ood_threshold

        # Basic config
        config.update(
            steps_per_epoch=5000,
            sac_kwargs=dict(batch_size=256,
                            agent_hidden=args.n_hidden,
                            gamma=0.99,
                            pi_lr=3e-4,
                            q_lr=3e-4,
                            ),
            rollouts_per_step=rollouts_per_step,
            max_rollout_length=max_rollout_length,
            model_pessimism=model_pessimism,
            ood_threshold=ood_threshold,
            pretrained_agent_path=args.pretrained_agent_path
        )

        # According to appendix in COMBO paper
        # if args.env_name in WALKER_ENVS:
        #     config['sac_kwargs'].update(pi_lr=1e-5, q_lr=1e-4)

        assert config['sac_kwargs']['batch_size'] is not None
        assert config['sac_kwargs']['agent_hidden'] is not None
        assert config['sac_kwargs']['gamma'] is not None
        assert config['sac_kwargs']['pi_lr'] is not None
        assert config['sac_kwargs']['q_lr'] is not None
        assert config['rollouts_per_step'] is not None
        assert config['max_rollout_length'] is not None
        assert config['model_pessimism'] is not None
        assert config['ood_threshold'] is not None

        ray.init()
        unfinished_jobs = []

        for seed in range(args.start_seed, args.start_seed+args.seeds):
            job_id = training_wrapper.remote(config, seed)
            unfinished_jobs.append(job_id)

        while unfinished_jobs:
            _, unfinished_jobs = ray.wait(unfinished_jobs)

    else:
        if args.level == 1:
            config.update(
                epochs=30,
                sac_kwargs=dict(batch_size=256,
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
                    "bounds": [1, 51],
                    "value_type": "int",
                    "log_scale": False,
                }]

            r_max, max_uncertainty, mean_uncertainty, std_uncertainty = get_uncertainty_distribution(args.env_name, args.mode)

            print("R_max: {}, Max uncertainty: {}, Mean uncertainty: {}".format(r_max, max_uncertainty, mean_uncertainty))

            if args.mode in PARTITIONING_MODES:
                parameters += [
                    {
                        "name": "ood_threshold",
                        "type": "range",
                        "bounds": [mean_uncertainty, max_uncertainty],
                        "value_type": "float",
                        "log_scale": True,
                    }]

                config.update(
                    model_pessimism=0
                )

            elif args.mode in PENALTY_MODES:
                parameters += [
                    {
                        "name": "model_pessimism",
                        "type": "range",
                        "bounds": [0, r_max/max_uncertainty],
                        "value_type": "float",
                        "log_scale": False,
                    }]

                config.update(
                    ood_threshold=0
                )

        assert config['sac_kwargs']['batch_size'] is not None
        assert config['sac_kwargs']['agent_hidden'] is not None
        assert config['sac_kwargs']['gamma'] is not None
        assert config['sac_kwargs']['pi_lr'] is not None
        assert config['sac_kwargs']['q_lr'] is not None

        ray.init()
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='avg_test_return',
            mode='max',
            max_t=10,
            grace_period=5,
            reduction_factor=3,
            brackets=1)

        search_alg = AxSearch(
            ax_client=AxClient(enforce_sequential_optimization=False),
            space=parameters,
            metric='avg_test_return',
            mode='max')

        analysis = tune.run(
            tune.with_parameters(training_function),
            name=args.env_name+'-'+config['mode'] +
            '-tuning-lvl-'+str(args.level),
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.n_trials,
            config=config,
            max_failures=3,
            resources_per_trial={"gpu": 0.5}
        )

        print("Best config: ", analysis.get_best_config(
            metric="avg_test_return", mode="max"))
