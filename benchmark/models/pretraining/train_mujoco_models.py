import torch
import gym
import argparse
from benchmark.user_config import MODELS_DIR
import os
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM, \
    HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_RANDOM, HOPPER_EXPERT, \
    HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_REPLAY, HOPPER_RANDOM, WALKER_EXPERT, \
    WALKER_MEDIUM, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_REPLAY, WALKER_RANDOM
from benchmark.utils.str2bool import str2bool
import d4rl  # noqa


def training_function(config, data, save_path=None, tuning=True):
    model = EnvironmentModel(hidden=4*[config['n_hidden']], **config)

    model.train_to_convergence(
        data=data, checkpoint_dir=None, tuning=tuning, **config)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)
        print("Saved model to: {}".format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default=WALKER_MEDIUM_REPLAY)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device != '':
        device = args.device

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
        "obs_bounds_trainable": True,
        "r_bounds_trainable": True,
        "patience": args.patience,
        "type": "probabilistic",
        "n_networks": 7,
        "pre_fn": pre_fn,
        "post_fn": post_fn,
        "debug": False,
        "no_reward": False,
        "use_batch_norm": False,
        "n_hidden": args.n_hidden,
        "lr": None,
        "batch_size": None,
        "in_normalized_space": True,
    }

    if args.level == 0:
        # Perform training with tuned hyperparameters and save model

        save_path = os.path.join(MODELS_DIR, args.env_name + '-model.pt')

        config.update(
            max_n_train_epochs=-1,
            debug=True)

        config.update(
            lr=args.lr,
            batch_size=256
        )

        assert config['lr'] is not None
        assert config['batch_size'] is not None

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
                batch_size=tune.choice([256, 512, 1024, 2048]))

        assert config['lr'] is not None
        assert config['batch_size'] is not None

        ray.init()
        scheduler = ASHAScheduler(
            metric='val_loss',
            mode='min',
            time_attr='time_since_restore',
            max_t=1000)

        search_alg = HyperOptSearch(
            metric='val_loss',
            mode='min')

        save_name = args.env_name

        if args.augment_loss:
            save_name += '-aug-loss'

        save_name += '-model-tuning-lvl-' + str(args.level)

        analysis = tune.run(
            tune.with_parameters(training_function, data=buffer),
            name=save_name,
            config=config,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=200,
            resources_per_trial={"gpu": 0.5},
            fail_fast=True
        )

        print("Best config: ", analysis.get_best_config(
            metric="val_loss", mode="min"))
