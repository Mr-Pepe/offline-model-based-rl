import argparse
from math import pi as PI

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.schemas import EnvironmentModelConfiguration
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.str2bool import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_points_between", type=str2bool, default=False)
    parser.add_argument("--steps_per_plot", type=int, default=100)
    args = parser.parse_args()

    steps = 50000
    steps_per_plot = args.steps_per_plot
    add_points_between = args.add_points_between
    plot = True
    device = "cpu"

    torch.manual_seed(0)

    x = torch.rand((1000,)) * PI - 2 * PI
    x = torch.cat((x, torch.rand((1000,)) * PI + PI))

    if add_points_between:
        x = torch.cat((x, torch.rand((100,)) * 0.5 - 0.25))

    x = x.to(device)

    y = torch.sin(x) + torch.normal(
        0, 0.225 * torch.abs(torch.sin(1.5 * x + PI / 8))
    ).to(device)

    n_networks = 2

    buffer = ReplayBuffer(1, 1, size=100000, device=device)
    buffer.obs_buf = x.unsqueeze(-1)
    buffer.next_obs_buf = y.unsqueeze(-1)
    buffer.rew_buf = y
    buffer.size = x.numel()

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        hidden_layer_sizes=[4, 8, 4],
        n_networks=n_networks,
        device=device,
        lr=1e-4,
        max_number_of_training_batches=steps_per_plot,
        training_batch_size=128,
    )

    model = EnvironmentModel(1, 1, model_config)

    x_true = torch.arange(-3 * PI, 3 * PI, 0.01)
    y_true = torch.sin(x_true)
    plt.figure()

    for _ in range(steps):
        model.train_to_convergence(buffer, config=model_config)

        if plot:
            _, mean_plt, logvar_plt, max_logvar_plt, _, = model(
                torch.cat(
                    (x_true.unsqueeze(-1), torch.zeros_like(x_true.unsqueeze(-1))),
                    dim=1,
                )
            )
            mean_plt = mean_plt[:, :, 1].detach().cpu()
            logvar_plt = logvar_plt[:, :, 1].detach().cpu()

            std = torch.exp(0.5 * logvar_plt)

            x_plt = x.cpu()
            y_plt = y.cpu()
            idx = list(range(0, len(x_plt), 10))

            plt.gca().clear()
            plt.scatter(x_plt[idx], y_plt[idx], color="green", marker="x", s=20)
            plt.plot(x_true, y_true, color="black")

            color = cm.rainbow(np.linspace(0, 1, n_networks))
            for i_network, c in zip(range(n_networks), color):

                plt.fill_between(
                    x_true,
                    (mean_plt[i_network] + std[i_network]).view(-1),
                    (mean_plt[i_network] - std[i_network]).view(-1),
                    color=c,
                    alpha=0.2,
                )
                plt.plot(x_true, mean_plt[i_network].view(-1), color=c)
                plt.ylim([-1.5, 1.5])
                plt.xlim([-8, 8])
                plt.xticks([])
                plt.yticks([])

            plt.draw()
            plt.pause(0.001)
