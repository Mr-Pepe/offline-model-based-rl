import argparse
import os
from pathlib import Path
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.str2bool import str2bool
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch


def training_function(config, data, checkpoint_dir=None, save_path=None):
    model = EnvironmentModel(hidden=4*[config['n_hidden']], **config)

    model.train_to_convergence(
        data=data, checkpoint_dir=checkpoint_dir, tuning=True, **config)

    if save_path:
        os.makedirs(os.path.dirname(save_path))
        torch.save(model, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='halfcheetah-medium-replay-v1')
    parser.add_argument('--final_training', type=str2bool, default=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device)

    pre_fn = get_preprocessing_function(args.env_name)
    assert pre_fn is not None
    post_fn = get_postprocessing_function(args.env_name)
    assert post_fn is not None

    if args.final_training:
        training_function(
            config={
                "max_n_train_epochs": -1,
                "patience": 20,
                "no_reward": False,
                "lr": 0.001359,
                "batch_size": 512,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_hidden": 512,
                "type": "probabilistic",
                "n_networks": 7,
                "pre_fn": pre_fn,
                "post_fn": post_fn,
                "debug": True,
                "use_batch_norm": True},
            data=buffer,
            save_path=os.path.join(
                str(Path.home()), 'Projects/thesis-code/data/models/cheetah/medium_replay.pt')
        )
    else:
        ray.init(local_mode=True)
        scheduler = ASHAScheduler(
            time_attr='time_since_restore',
            metric='val_loss',
            mode='min',
            max_t=1000,
            grace_period=5,
            reduction_factor=3,
            brackets=1)

        search_alg = HyperOptSearch(
            metric='val_loss',
            mode='min')

        analysis = tune.run(
            tune.with_parameters(training_function, data=buffer),
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=50,
            config={
                "max_n_train_epochs": 50,
                "patience": 30,
                "no_reward": False,
                "lr": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([256, 512, 1024, 2048]),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_hidden": tune.choice([64, 128, 256, 512]),
                "type": "probabilistic",
                "n_networks": 3,
                "pre_fn": pre_fn,
                "post_fn": post_fn,
                "debug": True,
                "use_batch_norm": tune.choice([True, False])
            },
            resources_per_trial={"gpu": 1}
        )

        print("Best config: ", analysis.get_best_config(
            metric="val_loss", mode="min"))
