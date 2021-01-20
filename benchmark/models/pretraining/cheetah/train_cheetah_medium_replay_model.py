import argparse
from ray import tune
import ray
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch


def training_function(config, data, checkpoint_dir=None):
    model = EnvironmentModel(hidden=4*[config['n_hidden']], **config)

    model.train_to_convergence(
        data=data, checkpoint_dir=checkpoint_dir, **config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='halfcheetah-medium-replay-v1')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device)

    pre_fn = get_preprocessing_function(args.env_name)
    assert pre_fn is not None
    post_fn = get_postprocessing_function(args.env_name)
    assert post_fn is not None

    ray.init(local_mode=True)
    scheduler = ASHAScheduler(
        time_attr='time_since_restore',
        metric='val_loss',
        mode='min',
        max_t=1000,
        grace_period=5,
        reduction_factor=3,
        brackets=1)

    analysis = tune.run(
        tune.with_parameters(training_function, data=buffer),
        scheduler=scheduler,
        num_samples=20,
        config={
            "max_n_train_epochs": 20,
            "patience": 20,
            "no_reward": False,
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([128, 256]),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "n_hidden": tune.choice([64, 128, 256, 512]),
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
