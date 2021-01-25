import argparse
from benchmark.user_config import MODELS_DIR
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
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM_REPLAY
import gym
import d4rl  # noqa
import torch


def training_function(config, data, checkpoint_dir=None, save_path=None, tuning=True):
    model = EnvironmentModel(hidden=4*[config['n_hidden']], **config)

    model.train_to_convergence(
        data=data, checkpoint_dir=checkpoint_dir, tuning=tuning, **config)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default=HALF_CHEETAH_MEDIUM_REPLAY)
    parser.add_argument('--level', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device)

    pre_fn = get_preprocessing_function(args.env_name)
    assert pre_fn is not None
    post_fn = get_postprocessing_function(args.env_name)
    assert post_fn is not None

    # None values must be filled for tuning and final training
    config = {
        "device": device,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "max_n_train_epochs": 50,
        "patience": 30,
        "type": "probabilistic",
        "n_networks": 7,
        "pre_fn": pre_fn,
        "post_fn": post_fn,
        "debug": False,
        "no_reward": False,
        "lr": None,
        "batch_size": None,
        "n_hidden": None,
        "use_batch_norm": None
    }

    if args.level == 0:
        # Perform training with tuned hyperparameters and save model

        save_path = os.path.join(MODELS_DIR, args.env_name+'-model')

        config.update(max_n_train_epochs=-1)

        if args.env_name == HALF_CHEETAH_MEDIUM_REPLAY:
            config.update(
                lr=0.003,
                batch_size=512,
                n_hidden=256,
                use_batch_norm=False)

        if args.env_name == HALF_CHEETAH_EXPERT:
            config.update(
                lr=0.002,
                batch_size=512,
                n_hidden=512,
                use_batch_norm=True)

        assert config['lr'] is not None
        assert config['batch_size'] is not None
        assert config['n_hidden'] is not None
        assert config['use_batch_norm'] is not None

        training_function(
            config=config,
            data=buffer,
            save_path=save_path,
            tuning=False
        )
    else:
        if args.level == 1:
            config.update(
                lr=tune.loguniform(1e-5, 1e-2),
                batch_size=tune.choice([256, 512, 1024, 2048]),
                n_hidden=tune.choice([64, 128, 256, 512]),
                use_batch_norm=tune.choice([True, False]))

        assert config['lr'] is not None
        assert config['batch_size'] is not None
        assert config['n_hidden'] is not None
        assert config['use_batch_norm'] is not None

        ray.init()
        scheduler = ASHAScheduler(
            metric='val_loss',
            mode='min',
            time_attr='time_since_restore',
            max_t=1000)

        search_alg = HyperOptSearch(
            metric='val_loss',
            mode='min')

        analysis = tune.run(
            tune.with_parameters(training_function, data=buffer),
            name=args.env_name + '-model-tuning-lvl-' + str(args.level),
            config=config,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=200,
            resources_per_trial={"gpu": 0.5},
            fail_fast=True
        )

        print("Best config: ", analysis.get_best_config(
            metric="val_loss", mode="min"))
