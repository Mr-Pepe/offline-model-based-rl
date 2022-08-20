import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns


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
trajectory_color = 'black'
virtual_rollout_color = 'coral'


points = [[x[i], y[i]] for (x, y) in trajectories for i in range(len(x))]
x = [point[0] for point in points]
y = [point[1] for point in points]

# Penalty
# sns.kdeplot(x, y, shade=True, bw_adjust=4, levels=100)
# sns.kdeplot(x, y, shade=True, bw_adjust=0.6,
#             levels=100, thresh=0.0000000000000001)

# Partitioning
sns.kdeplot(x, y, shade=True, bw_adjust=0.6,
            levels=2, thresh=0.4, color='blue')
# sns.kdeplot(x, y, shade=True, bw_adjust=0.6,
#             levels=2, thresh=0.02, color='blue')


for i_rollout in range(n_virtual_rollouts):
    x, y = points[random.randint(0, len(points) - 1)]

    for step in range(rollout_length):
        u = random.random()*max_step_length-0.5*max_step_length
        v = random.random()*max_step_length-0.5*max_step_length

        plt.quiver(x, y, u, v, angles='xy', scale=1, scale_units='xy',
                   color=virtual_rollout_color, width=0.004)

        x += u
        y += v

# Plot trajectories
for x, y in trajectories:
    plt.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y), angles='xy',
               scale=1, scale_units='xy', color=trajectory_color, width=0.005)

    plt.scatter(x, y, color=trajectory_color, s=50)


plt.axis('off')
plt.xlim([-1, 4])
plt.ylim([-1, 4])
plt.tight_layout()
plt.show()
