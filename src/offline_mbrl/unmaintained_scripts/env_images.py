import os.path as osp
from pathlib import Path

# pylint: disable=unused-import
import gym
import matplotlib.pyplot as plt

from offline_mbrl.utils.envs import (
    HALF_CHEETAH_MEDIUM_REPLAY_V2,
    HOPPER_MEDIUM_REPLAY_V2,
    WALKER_MEDIUM_REPLAY_V2,
)

if __name__ == "__main__":
    seeds = [53, 71, 123]
    envs = [
        gym.make(HALF_CHEETAH_MEDIUM_REPLAY_V2),
        gym.make(HOPPER_MEDIUM_REPLAY_V2),
        gym.make(WALKER_MEDIUM_REPLAY_V2),
    ]
    images = []

    for i, env in enumerate(envs):
        env.reset(seed=seeds[i])
        env.action_space.seed(seeds[i])

        for step in range(30):
            obs, _, _, _ = env.step(env.action_space.sample())

        images.append(env.render(mode="rgb_array"))

    f, axes = plt.subplots(1, 3, figsize=(10, 10))

    x_labels = ["Halfcheetah", "Hopper", "Walker2d"]

    for i, image in enumerate(images):
        ax = axes[i]
        ax.clear()
        ax.imshow(image)
        ax.set_xlabel(x_labels[i], fontsize=24, labelpad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

    f.savefig(osp.join(Path.home(), "tmp", "environments.pdf"), bbox_inches="tight")
    plt.show()
