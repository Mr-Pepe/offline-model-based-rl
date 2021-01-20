import argparse
from ray import tune
import ray
from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch


def training_function(config, checkpoint_dir=None):
    model = EnvironmentModel(**config)

    model.train_to_convergence(**config)


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
    analysis = tune.run(
        training_function,
        metric="val_loss",
        mode="min",
        num_samples=10,
        config={
            "max_n_train_epochs": 3,
            "data": buffer,
            "patience": 20,
            "no_reward": False,
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden": tune.choice([64, 64, 64, 64],
                                  [128, 128, 128, 128],
                                  [256, 256, 256, 256],
                                  [512, 512, 512, 512]),
            "n_networks": 1,
            "pre_fn": pre_fn,
            "post_fn": post_fn,
            "debug": True,
            "use_batch_norm": tune.choice([True, False])
        },
        resources_per_trial={"gpu": 1}
    )

    print("Best config: ", analysis.get_best_config(
        metric="val_loss", mode="min"))
