import matplotlib.pyplot as plt
import gym
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.mazes import \
    plot_antmaze_umaze_walls, \
    plot_maze2d_umaze_walls
import d4rl  # noqa


def visualize_antmaze_umaze():
    env = gym.make('antmaze-umaze-v0')
    buffer, _, _ = load_dataset_from_env(env, n_samples=950000)

    print("Samples: {}".format(buffer.size))

    plt.scatter(buffer.obs_buf[buffer.done_buf == False, 0],
                buffer.obs_buf[buffer.done_buf == False, 1],
                marker='.',
                s=2,
                zorder=2)
    plt.scatter(buffer.obs_buf[buffer.done_buf == True, 0],
                buffer.obs_buf[buffer.done_buf == True, 1],
                marker='.',
                s=50,
                zorder=3)
    plot_antmaze_umaze_walls()
    plt.show()


def visualize_maze2d_umaze():
    env = gym.make('maze2d-umaze-v1')
    buffer, _, _ = load_dataset_from_env(env, n_samples=95000)

    print("Samples: {}".format(buffer.size))

    plt.scatter(buffer.obs_buf[buffer.done_buf == False, 0],
                buffer.obs_buf[buffer.done_buf == False, 1],
                marker='.',
                s=2,
                zorder=2)
    plt.scatter(buffer.obs_buf[buffer.done_buf == True, 0],
                buffer.obs_buf[buffer.done_buf == True, 1],
                marker='.',
                s=50,
                zorder=3)
    plot_maze2d_umaze_walls()
    plt.show()


visualize_maze2d_umaze()
