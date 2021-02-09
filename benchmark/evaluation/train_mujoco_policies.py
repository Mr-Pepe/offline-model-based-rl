import argparse
from benchmark.utils.run_utils import setup_logger_kwargs
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from benchmark.utils.envs import \
    HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_REPLAY, \
    HOPPER_MEDIUM_EXPERT, WALKER_MEDIUM_EXPERT
from benchmark.user_config import MODELS_DIR
from benchmark.train import Trainer
from benchmark.utils.str2bool import str2bool
import torch
import os


def training_function(config, tuning=True):
    config["sac_kwargs"].update(
        {"hidden": 4*[config["sac_kwargs"]["agent_hidden"]]})
    trainer = Trainer(**config)
    trainer.train(tuning=tuning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default=HALF_CHEETAH_MEDIUM_REPLAY)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--mode', type=str, default='mopo')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--pessimism', type=float, default=1)
    parser.add_argument('--ood_threshold', type=float, default=0.5)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--rollout_length', type=int, default=1)
    parser.add_argument('--pretrained_agent_path', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device != '':
        device = args.device

    pretrained_model_name = args.env_name + '-model.pt'

    # None values must be filled for tuning and final training
    config = dict(
        env_name=args.env_name,
        sac_kwargs=dict(batch_size=None,
                        agent_hidden=None,
                        gamma=None,
                        pi_lr=None,
                        q_lr=None,
                        ),
        rollouts_per_step=None,
        max_rollout_length=None,
        model_pessimism=None,
        ood_threshold=None,
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
        real_buffer_size=int(1e6),
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

        # Basic config
        config.update(
            steps_per_epoch=4000,
            sac_kwargs=dict(batch_size=256,
                            agent_hidden=128,
                            gamma=0.99,
                            pi_lr=3e-4,
                            q_lr=3e-4,
                            ),
            rollouts_per_step=100,
            max_rollout_length=args.rollout_length,
            model_pessimism=args.pessimism,
            ood_threshold=args.ood_threshold,
            pretrained_agent_path=args.pretrained_agent_path
        )

        if args.env_name in [HALF_CHEETAH_MEDIUM_EXPERT,
                             HOPPER_MEDIUM_EXPERT,
                             WALKER_MEDIUM_EXPERT]:
            config.update(real_buffer_size=int(2e6))

        assert config['sac_kwargs']['batch_size'] is not None
        assert config['sac_kwargs']['agent_hidden'] is not None
        assert config['sac_kwargs']['gamma'] is not None
        assert config['sac_kwargs']['pi_lr'] is not None
        assert config['sac_kwargs']['q_lr'] is not None
        assert config['rollouts_per_step'] is not None
        assert config['max_rollout_length'] is not None
        assert config['model_pessimism'] is not None
        assert config['ood_threshold'] is not None

        for seed in range(args.start_seed, args.start_seed+args.seeds):
            exp_name = args.env_name+'-' + \
                config['mode']+'-' + str(config['max_rollout_length'])+'steps'

            if config['mode'] == 'mopo':
                exp_name += '-' + str(config['model_pessimism']) + 'pessimism'

            if config['mode'] == 'morel':
                exp_name += '-' + str(config['ood_threshold']) + 'threshold'

            if config['mode'] == 'pepe':
                exp_name += '-' + str(config['ood_threshold']) + 'threshold'

            config.update(
                epochs=args.epochs,
                seed=seed,
                logger_kwargs=setup_logger_kwargs(exp_name,
                                                  seed=seed),
            )
            training_function(config, tuning=False)

    else:
        if args.level == 1:
            config.update(
                epochs=30,
                sac_kwargs=dict(batch_size=tune.choice([128, 256, 512]),
                                agent_hidden=tune.choice([32, 64, 128, 256]),
                                gamma=tune.uniform(0.99, 0.999),
                                pi_lr=tune.loguniform(1e-5, 1e-2),
                                q_lr=tune.loguniform(1e-5, 1e-2),
                                ),
                rollouts_per_step=tune.randint(1, 401),
                max_rollout_length=tune.randint(1, 50),
                model_pessimism=tune.uniform(0.001, 1000)
            )

            if args.env_name == HALF_CHEETAH_MEDIUM_EXPERT:
                config.update(real_buffer_size=int(2e6))

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
        scheduler = ASHAScheduler(
            time_attr='time_since_restore',
            metric='avg_test_return',
            mode='max',
            max_t=1000,
            grace_period=10,
            reduction_factor=3,
            brackets=1)

        search_alg = HyperOptSearch(
            metric='avg_test_return',
            mode='max')

        analysis = tune.run(
            tune.with_parameters(training_function),
            name=args.env_name+'-'+config['mode'] +
            '-tuning-lvl-'+str(args.level),
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=200,
            config=config,
            resources_per_trial={"gpu": 0.5}
        )

        print("Best config: ", analysis.get_best_config(
            metric="avg_test_return", mode="max"))
