from benchmark.utils.replay_buffer import ReplayBuffer
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.mazes import plot_antmaze_umaze
import matplotlib.pyplot as plt
import torch


def augment_antmaze_umaze_dataset(f_path, plot=False):
    env = gym.make('antmaze-umaze-v0')
    original_buffer, obs_dim, act_dim = load_dataset_from_env(env)

    if plot:
        plot_antmaze_umaze(buffer=original_buffer)
        plt.show()

    augmented_buffer = ReplayBuffer(obs_dim,
                                    act_dim,
                                    2*original_buffer.size)

    # Take different parts of the dataset based on x/y position and shift them
    # to other positions
    augmentations = [
        ((original_buffer.obs_buf[:, 1] > 6.5), (0, -8)),
        ((original_buffer.obs_buf[:, 1] < 1.5), (0, 8)),
    ]

    for augmentation in augmentations:

        idx = augmentation[0]
        x_shift = augmentation[1][0]
        y_shift = augmentation[1][1]

        augmented_obs = original_buffer.obs_buf[idx]
        augmented_next_obs = original_buffer.obs2_buf[idx]
        augmented_obs[:, 0] += x_shift
        augmented_next_obs[:, 0] += x_shift
        augmented_obs[:, 1] += y_shift
        augmented_next_obs[:, 1] += y_shift

        augmented_buffer.store_batch(augmented_obs,
                                     original_buffer.act_buf[idx],
                                     torch.zeros(((idx > 0).sum())),
                                     augmented_next_obs,
                                     torch.zeros(((idx > 0).sum())))

        augmented_buffer.obs_buf

        if plot:
            plot_antmaze_umaze(buffer=augmented_buffer)
            plt.show()

    augmented_buffer.store_batch(
        original_buffer.obs_buf,
        original_buffer.act_buf,
        original_buffer.rew_buf,
        original_buffer.obs2_buf,
        original_buffer.done_buf
    )

    if plot:
        plot_antmaze_umaze(buffer=augmented_buffer)
        plt.show()

    torch.save(augmented_buffer, f_path)


augment_antmaze_umaze_dataset(
    '/home/felipe/Projects/thesis-code/data/datasets/antmaze_umaze_augmented.p')
