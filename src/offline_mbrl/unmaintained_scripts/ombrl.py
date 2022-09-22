import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    random.seed(5)

    trajectories = [
        [[0.4, 1.5, 2.2, 2, 3], [0, 0.4, 1, 1.6, 2.2]],
        [[0, 0.3, 1, 2, 2.5], [0.3, 1.2, 2, 2.1, 2.9]],
        [[0, 1.2, 1.8, 3], [0, 1.1, 1.9, 2.7]],
        [[0.5, 1.8, 2.1, 2.8], [0.6, 1.2, 1.9, 2.6]],
    ]

    n_virtual_rollouts = 20
    rollout_length = 2
    max_step_length = 1.5
    trajectory_color = "black"
    virtual_rollout_color = "coral"

    points: list[tuple[float, float]] = [
        (x[i], y[i]) for (x, y) in trajectories for i in range(len(x))
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    # Penalty
    # sns.kdeplot(x, y, shade=True, bw_adjust=4, levels=100)
    # sns.kdeplot(x, y, shade=True, bw_adjust=0.6,
    #             levels=100, thresh=0.0000000000000001)

    # Partitioning
    sns.kdeplot(xs, ys, shade=True, bw_adjust=0.6, levels=2, thresh=0.4, color="blue")
    # sns.kdeplot(x, y, shade=True, bw_adjust=0.6,
    #             levels=2, thresh=0.02, color='blue')

    for i_rollout in range(n_virtual_rollouts):
        point = points[random.randint(0, len(points) - 1)]
        x, y = point

        for step in range(rollout_length):
            u = random.random() * max_step_length - 0.5 * max_step_length
            v = random.random() * max_step_length - 0.5 * max_step_length

            plt.quiver(
                x,
                y,
                u,
                v,
                angles="xy",
                scale=1,
                scale_units="xy",
                color=virtual_rollout_color,
                width=0.004,
            )

            x += u
            y += v

    # Plot trajectories
    for xs, ys in trajectories:
        plt.quiver(
            xs[:-1],
            ys[:-1],
            np.diff(xs),
            np.diff(ys),
            angles="xy",
            scale=1,
            scale_units="xy",
            color=trajectory_color,
            width=0.005,
        )

        plt.scatter(xs, ys, color=trajectory_color, s=50)

    plt.axis("off")
    plt.xlim([-1, 4])
    plt.ylim([-1, 4])
    plt.tight_layout()
    plt.show()
