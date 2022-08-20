import d4rl  # noqa
import gym
import matplotlib.pyplot as plt
import torch
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.mazes import plot_antmaze_medium, plot_antmaze_umaze
from offline_mbrl.utils.postprocessing import postprocess_antmaze_medium
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.reward_functions import antmaze_medium_diverse_rew_fn


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


def augment_antmaze_medium_dataset(buffer, f_path, plot=False):
    augmented_buffer = ReplayBuffer(obs_dim,
                                    act_dim,
                                    5000000)

    # Define where to shift blocks of the environment
    goal_blocks = [(-2, -2),
                   (-2, 2),
                   (2, -2),
                   (2, 2),
                   (2, 6),
                   (6, 6),
                   (10, 6),
                   (10, 10),
                   (14, 10),
                   (18, 10),
                   (18, 14),
                   (18, 18)]

    origin_blocks = [(-2, -2),
                     (-2, 2),
                     (-2, 10),
                     (-2, 14),
                     (-2, 18),
                     (2, -2),
                     (2, 2),
                     (2, 6),
                     (2, 10),
                     (2, 18),
                     (6, 6),
                     (6, 14),
                     (6, 18),
                     (10, 2),
                     (10, 6),
                     (10, 10),
                     (10, 14),
                     (14, -2),
                     (14, 2),
                     (14, 10),
                     (14, 18),
                     (18, -2),
                     (18, 2),
                     (18, 10),
                     (18, 14),
                     (18, 18)]

    x = buffer.obs_buf[:buffer.size, 0]
    y = buffer.obs_buf[:buffer.size, 1]

    augmented_samples = 0

    for origin_block in origin_blocks:

        for goal_block in goal_blocks:
            if goal_block == origin_block:
                ant_radius = 0
            else:
                ant_radius = 0.8

            idx = ((origin_block[0] + ant_radius <= x) *
                   (x <= origin_block[0] + 4 - ant_radius) *
                   (origin_block[1] + ant_radius <= y) *
                   (y <= origin_block[1] + 4 - ant_radius))
            idx = torch.cat(
                (idx, torch.zeros((buffer.max_size - buffer.size), dtype=torch.bool)))

            x_shift = goal_block[0] - origin_block[0]
            y_shift = goal_block[1] - origin_block[1]

            augmented_obs = buffer.obs_buf[idx]
            augmented_next_obs = buffer.obs2_buf[idx]
            augmented_obs[:, 0] += x_shift
            augmented_next_obs[:, 0] += x_shift
            augmented_obs[:, 1] += y_shift
            augmented_next_obs[:, 1] += y_shift

            augmented_buffer.store_batch(augmented_obs,
                                         buffer.act_buf[idx],
                                         torch.zeros(((idx > 0).sum())),
                                         augmented_next_obs,
                                         torch.zeros(((idx > 0).sum())))

            augmented_samples += len(augmented_obs)

            print("Augmented samples: {}".format(augmented_samples))

    if augmented_samples > augmented_buffer.max_size:
        print("Augmented buffer too small. Should at least have size: {}".format(
            augmented_samples))

    augmented_buffer.rew_buf = antmaze_medium_diverse_rew_fn(
        augmented_buffer.obs2_buf.unsqueeze(0)).view(-1)

    if plot:
        plot_antmaze_medium(buffer=augmented_buffer, n_samples=20000)
        plt.show()

    torch.save(augmented_buffer, f_path)


def antmaze_augmentation(obs, obs2, xmin, xmax, ymin, ymax):
    n_samples = len(obs)
    x = torch.rand((n_samples,), device=obs.device)*(xmax-xmin)+xmin
    y = torch.rand((n_samples,), device=obs.device)*(ymax-ymin)+ymin

    x_shift = x - obs[:, 0]
    y_shift = y - obs[:, 1]

    obs[:, 0] += x_shift
    obs2[:, 0] += x_shift
    obs[:, 1] += y_shift
    obs2[:, 1] += y_shift


if __name__ == "__main__":
    env = gym.make('antmaze-medium-diverse-v0')
    ant_radius = 0.8

    buffer, obs_dim, act_dim = load_dataset_from_env(env)
    train_buffer = ReplayBuffer(
        obs_dim, act_dim, buffer.size)
    train_idx = ~postprocess_antmaze_medium(
        next_obs=buffer.obs2_buf[:buffer.size].unsqueeze(0), ant_radius=ant_radius)['dones'].view(-1)
    train_idx = torch.logical_and(train_idx, (buffer.obs2_buf[:, 2] > 0.3))

    train_buffer.store_batch(buffer.obs_buf[train_idx],
                             buffer.act_buf[train_idx],
                             buffer.rew_buf[train_idx],
                             buffer.obs2_buf[train_idx],
                             buffer.done_buf[train_idx],)

    buffer = train_buffer
    augment_antmaze_medium_dataset(buffer, '')
