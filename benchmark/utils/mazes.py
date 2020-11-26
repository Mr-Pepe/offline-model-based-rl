import matplotlib.pyplot as plt
import torch


def get_umaze_walls():
    walls = []
    for i in range(len(U_MAZE_STRUCTURE)):
        for j in range(len(U_MAZE_STRUCTURE[0])):
            if U_MAZE_STRUCTURE[i][j] == 1:
                minx = j * U_MAZE_SCALING - \
                    U_MAZE_SCALING * 0.5 + U_MAZE_OFFSET[0]
                maxx = j * U_MAZE_SCALING + \
                    U_MAZE_SCALING * 0.5 + U_MAZE_OFFSET[0]
                miny = i * U_MAZE_SCALING - \
                    U_MAZE_SCALING * 0.5 + U_MAZE_OFFSET[1]
                maxy = i * U_MAZE_SCALING + \
                    U_MAZE_SCALING * 0.5 + U_MAZE_OFFSET[1]

                walls.append([minx, maxx, miny, maxy])

    return torch.as_tensor(walls)


# From https://github.com/rail-berkeley/d4rl/blob/master/d4rl/locomotion/maze_env.py
U_MAZE_STRUCTURE = [[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]


# From https://github.com/rail-berkeley/d4rl/blob/master/d4rl/locomotion/__init__.py

U_MAZE_SCALING = 4
U_MAZE_OFFSET = [-4, -4]
U_MAZE_WALLS = get_umaze_walls()


def plot_umaze_walls():
    walls = get_umaze_walls()

    for wall in walls:
        plt.fill([wall[0], wall[0], wall[1], wall[1], wall[0]],
                 [wall[2], wall[3], wall[3], wall[2], wall[2]],
                 color='grey', zorder=-1)

    plt.xlim([-3, 12])
    plt.ylim([-3, 12])
