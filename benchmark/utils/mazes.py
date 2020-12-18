import matplotlib.pyplot as plt
import torch


def get_antmaze_umaze_walls(device='cpu'):
    walls = []
    for i in range(len(ANTMAZE_UMAZE_STRUCTURE)):
        for j in range(len(ANTMAZE_UMAZE_STRUCTURE[0])):
            if ANTMAZE_UMAZE_STRUCTURE[i][j] == 1:
                minx = j * ANTMAZE_UMAZE_SCALING - \
                    ANTMAZE_UMAZE_SCALING * 0.5 + ANTMAZE_UMAZE_OFFSET[0]
                maxx = j * ANTMAZE_UMAZE_SCALING + \
                    ANTMAZE_UMAZE_SCALING * 0.5 + ANTMAZE_UMAZE_OFFSET[0]
                miny = i * ANTMAZE_UMAZE_SCALING - \
                    ANTMAZE_UMAZE_SCALING * 0.5 + ANTMAZE_UMAZE_OFFSET[1]
                maxy = i * ANTMAZE_UMAZE_SCALING + \
                    ANTMAZE_UMAZE_SCALING * 0.5 + ANTMAZE_UMAZE_OFFSET[1]

                walls.append([minx, maxx, miny, maxy])

    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    return torch.as_tensor(walls, device=device)


def get_antmaze_medium_walls(device='cpu'):
    walls = []
    for i in range(len(ANTMAZE_MEDIUM_STRUCTURE)):
        for j in range(len(ANTMAZE_MEDIUM_STRUCTURE[0])):
            if ANTMAZE_MEDIUM_STRUCTURE[i][j] == 1:
                minx = j * ANTMAZE_MEDIUM_SCALING - \
                    ANTMAZE_MEDIUM_SCALING * 0.5 + ANTMAZE_MEDIUM_OFFSET[0]
                maxx = j * ANTMAZE_MEDIUM_SCALING + \
                    ANTMAZE_MEDIUM_SCALING * 0.5 + ANTMAZE_MEDIUM_OFFSET[0]
                miny = i * ANTMAZE_MEDIUM_SCALING - \
                    ANTMAZE_MEDIUM_SCALING * 0.5 + ANTMAZE_MEDIUM_OFFSET[1]
                maxy = i * ANTMAZE_MEDIUM_SCALING + \
                    ANTMAZE_MEDIUM_SCALING * 0.5 + ANTMAZE_MEDIUM_OFFSET[1]

                walls.append([minx, maxx, miny, maxy])

    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    return torch.as_tensor(walls, device=device)


def get_maze2d_umaze_walls():
    walls = []
    for i in range(len(MAZE2D_UMAZE_STRUCTURE)):
        for j in range(len(MAZE2D_UMAZE_STRUCTURE[0])):
            if MAZE2D_UMAZE_STRUCTURE[i][j] == 1:
                minx = j * MAZE2D_UMAZE_SCALING - \
                    MAZE2D_UMAZE_SCALING*0.5 + MAZE2D_UMAZE_OFFSET[0]
                maxx = j * MAZE2D_UMAZE_SCALING + \
                    MAZE2D_UMAZE_SCALING*0.5 + MAZE2D_UMAZE_OFFSET[0]
                miny = i * MAZE2D_UMAZE_SCALING - \
                    MAZE2D_UMAZE_SCALING*0.5 + MAZE2D_UMAZE_OFFSET[1]
                maxy = i * MAZE2D_UMAZE_SCALING + \
                    MAZE2D_UMAZE_SCALING*0.5 + MAZE2D_UMAZE_OFFSET[1]

                walls.append([minx, maxx, miny, maxy])

    return torch.as_tensor(walls)


def plot_antmaze_umaze(xlim=None, ylim=None, buffer=None):
    walls = get_antmaze_umaze_walls()

    for wall in walls:
        plt.fill([wall[0], wall[0], wall[1], wall[1], wall[0]],
                 [wall[2], wall[3], wall[3], wall[2], wall[2]],
                 color='grey', zorder=-1)

    if buffer:
        if buffer.size > 10000:
            idx = torch.randint(0, buffer.size, (10000,))
        else:
            idx = torch.arange(buffer.size)

        rewards = buffer.rew_buf[:buffer.size].cpu()[idx]
        dones = buffer.done_buf[:buffer.size].cpu()[idx]
        next_obs = buffer.obs2_buf[:buffer.size].cpu()[idx]

        plt.scatter(
            next_obs[dones, 0].cpu(),
            next_obs[dones, 1].cpu(),
            color='red',
            marker='.',
            s=5
        )
        plt.scatter(
            next_obs[~dones, 0].cpu(),
            next_obs[~dones, 1].cpu(),
            c=rewards[~dones],
            cmap='cividis',
            marker='.',
            s=2
        )
        plt.colorbar()

    if xlim is None:
        plt.xlim([-3, 12])
    else:
        plt.xlim(xlim)

    if ylim is None:
        plt.ylim([-3, 12])
    else:
        plt.ylim(ylim)


def plot_antmaze_medium(xlim=None, ylim=None, buffer=None):
    walls = get_antmaze_medium_walls()

    for wall in walls:
        plt.fill([wall[0], wall[0], wall[1], wall[1], wall[0]],
                 [wall[2], wall[3], wall[3], wall[2], wall[2]],
                 color='grey', zorder=-1)

    if buffer:
        # if buffer.size > 10000:
        #     idx = torch.randint(0, buffer.size, (10000,))
        # else:
        idx = torch.arange(buffer.size)

        rewards = buffer.rew_buf[:buffer.size].cpu()[idx]
        dones = buffer.done_buf[:buffer.size].cpu()[idx]
        next_obs = buffer.obs2_buf[:buffer.size].cpu()[idx]

        plt.scatter(
            next_obs[dones, 0].cpu(),
            next_obs[dones, 1].cpu(),
            color='red',
            marker='.',
            s=5
        )
        plt.scatter(
            next_obs[~dones, 0].cpu(),
            next_obs[~dones, 1].cpu(),
            c=rewards[~dones],
            cmap='cividis',
            marker='.',
            s=2
        )
        plt.colorbar()

    if xlim is None:
        plt.xlim([-5, 25])
    else:
        plt.xlim(xlim)

    if ylim is None:
        plt.ylim([-5, 25])
    else:
        plt.ylim(ylim)


def plot_maze2d_umaze(xlim=None, ylim=None, buffer=None):
    walls = get_maze2d_umaze_walls()

    for wall in walls:
        plt.fill([wall[0], wall[0], wall[1], wall[1], wall[0]],
                 [wall[2], wall[3], wall[3], wall[2], wall[2]],
                 color='grey', zorder=-1)

    if buffer:
        rewards = buffer.rew_buf[:buffer.size].cpu()
        plt.scatter(
            buffer.obs2_buf[:buffer.size, 0].cpu(),
            buffer.obs2_buf[:buffer.size, 1].cpu(),
            c=rewards,
            cmap='cividis',
            marker='.',
            s=2
        )
        plt.colorbar()

    if xlim is None:
        plt.xlim([0, 3.5])
    else:
        plt.xlim(xlim)

    if ylim is None:
        plt.ylim([0, 3.5])
    else:
        plt.ylim(ylim)


# From https://github.com/rail-berkeley/d4rl/blob/master/d4rl/locomotion/maze_env.py
ANTMAZE_UMAZE_STRUCTURE = [[1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 1],
                           [1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1]]

ANTMAZE_MEDIUM_STRUCTURE = [[1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 1, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1]]


# From https://github.com/rail-berkeley/d4rl/blob/master/d4rl/locomotion/__init__.py

ANTMAZE_UMAZE_SCALING = 4
ANTMAZE_UMAZE_OFFSET = [-4, -4]
ANTMAZE_UMAZE_WALLS = get_antmaze_umaze_walls()
ANTMAZE_UMAZE_WALLS_CUDA = get_antmaze_umaze_walls(device='cuda')
ANTMAZE_UMAZE_GOAL_BLOCK = [-2, 2, 8, 10]

ANTMAZE_MEDIUM_SCALING = 4
ANTMAZE_MEDIUM_OFFSET = [-4, -4]
ANTMAZE_MEDIUM_WALLS = get_antmaze_medium_walls()
ANTMAZE_MEDIUM_WALLS_CUDA = get_antmaze_medium_walls(device='cuda')
ANTMAZE_MEDIUM_GOAL_BLOCK = [-2, 2, 8, 10]

ANTMAZE_ANT_RADIUS = 0.4


MAZE2D_UMAZE_STRUCTURE = [[1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1]]

MAZE2D_UMAZE_OFFSET = [-0.2, -0.2]
MAZE2D_UMAZE_SCALING = 1
MAZE2D_POINT_RADIUS = 0.12

MAZE2D_UMAZE_WALLS = get_maze2d_umaze_walls()
