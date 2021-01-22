import argparse
from benchmark.utils.run_utils import setup_logger_kwargs
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from benchmark.utils.str2bool import str2bool
from benchmark.utils.envs import HALF_CHEETAH_MEDIUM_REPLAY
from benchmark.train import Trainer
import torch


def training_function(config, tuning=True):
    config["sac_kwargs"].update(
        {"hidden": 4*[config["sac_kwargs"]["agent_hidden"]]})
    trainer = Trainer(**config)
    trainer.train(tuning=tuning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default=HALF_CHEETAH_MEDIUM_REPLAY)
    parser.add_argument('--level', type=str2bool, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # This is the optimal configuration that has been found so far.
    # Certain values get replaced during tuning
    config = dict(
        env_name=args.env_name,
        sac_kwargs=dict(batch_size=128,
                        agent_hidden=64,
                        gamma=0.996,
                        pi_lr=7e-5,
                        q_lr=6e-4,
                        ),
        model_kwargs=dict(),
        dataset_path='',
        seed=0,
        epochs=30,
        steps_per_epoch=1000,
        random_steps=3000,
        init_steps=1000,
        env_steps_per_step=0,
        n_samples_from_dataset=-1,
        agent_updates_per_step=1,
        num_test_episodes=10,
        curriculum=[1, 1, 20, 100],
        max_ep_len=1000,
        use_model=True,
        pretrained_agent_path='',
        pretrained_model_path='/home/felipe/Projects/thesis-code/data/models/cheetah/medium_replay.pt',
        model_pessimism=50,
        ood_threshold=-1,
        exploration_mode='reward',
        uncertainty='aleatoric',
        model_max_n_train_batches=-1,
        rollouts_per_step=38,
        rollout_schedule=[1, 1, 20, 100],
        max_rollout_length=8,
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
        config.update(
            epochs=200,
            logger_kwargs=setup_logger_kwargs('cheetah_medium_replay_mopo')
        )
        training_function(config, tuning=False)

    else:
        # Tuning gets increasingly specific
        if args.level == 1:
            max_t = 1000
            num_samples = 100
            config.update(
                epochs=30,
                sac_kwargs=dict(batch_size=tune.choice([128, 256, 512]),
                                agent_hidden=tune.choice([32, 64, 128, 256]),
                                gamma=tune.uniform(0.99, 0.999),
                                pi_lr=tune.loguniform(1e-5, 1e-3),
                                q_lr=tune.loguniform(1e-5, 1e-3),
                                ),
                model_pessimism=tune.uniform(0, 50),
                rollouts_per_step=tune.randint(1, 401),
                max_rollout_length=tune.randint(1, 10),
            )
        else:
            raise ValueError("Not tuning level {}".format(args.level))

        ray.init(local_mode=True)
        scheduler = ASHAScheduler(
            time_attr='time_since_restore',
            metric='avg_test_return',
            mode='max',
            max_t=max_t,
            grace_period=10,
            reduction_factor=3,
            brackets=1)

        search_alg = HyperOptSearch(
            metric='avg_test_return',
            mode='max')

        analysis = tune.run(
            tune.with_parameters(training_function),
            name=args.env_name+'-mopo-tuning-lvl-'+args.level,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            config=config,
            resources_per_trial={"gpu": 1}
        )

        print("Best config: ", analysis.get_best_config(
            metric="avg_test_return", mode="max"))
