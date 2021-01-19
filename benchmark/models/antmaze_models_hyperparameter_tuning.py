import argparse
from ray import tune
from benchmark.utils.mazes import ANTMAZE_MEDIUM_MAX, ANTMAZE_MEDIUM_MIN
from benchmark.utils.augmentation import antmaze_augmentation
from benchmark.utils.replay_buffer import ReplayBuffer
import os
from benchmark.utils.postprocessing import get_postprocessing_function, postprocess_antmaze_medium
from benchmark.utils.preprocessing import get_preprocessing_function, preprocess_antmaze_medium
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch


def training_function(config, checkpoint_dir=None):
    # tune.util.wait_for_gpu()
    model = EnvironmentModel(hidden=[config['l1'],
                                     config['l2'],
                                     config['l3'],
                                     config['l4']],
                             **config)

    model.train_to_convergence(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-v0')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device='cpu')

    analysis = tune.run(
        training_function,
        config={
            "max_n_train_epochs": 3,
            "num_samples": 10,
            "metric": "val_loss",
            "mode": "min",
            "data": buffer,
            "patience": 20,
            "no_reward": True,
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "l1": tune.randint(16, 513),
            "l2": tune.randint(16, 513),
            "l3": tune.randint(16, 513),
            "l4": tune.randint(16, 513),
            "type": "probabilistic",
            "n_networks": 1,
            "pre_fn": preprocess_antmaze_medium,
            "post_fn": postprocess_antmaze_medium,
        },
        resources_per_trial={"gpu": 1}
    )

    print("Best config: ", analysis.get_best_config(
        metric="val_loss", mode="min"))
