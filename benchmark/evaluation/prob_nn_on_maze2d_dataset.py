import torch
from torch.optim.adam import Adam
from benchmark.models.environment_model import EnvironmentModel
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.mazes import plot_maze2d_umaze


def test_probabilistic_model_trains_on_maze2d_umaze(steps=3000, plot=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_samples = 100
    train_idx = torch.ones((n_samples), dtype=torch.bool)
    frac = int(0.3*n_samples)
    train_idx[frac:-frac] = 0

    torch.manual_seed(0)

    env = gym.make('maze2d-umaze-v1')
    buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples=n_samples,
                                                     buffer_device=device)
    model = EnvironmentModel(
        obs_dim, act_dim,
        hidden=[64, 64, 64], type='probabilistic', device=device)

    lr = 1e-4
    optim = Adam(model.parameters(), lr=lr)

    loss = torch.tensor(0)

    f = plt.figure()

    for i in range(steps):

        x = torch.cat((buffer.obs_buf[train_idx],
                       buffer.act_buf[train_idx]), dim=1)
        y = torch.cat((buffer.obs2_buf[train_idx],
                       buffer.rew_buf[train_idx].unsqueeze(1)), dim=1)

        x = x.to(device)
        y = y.to(device)

        optim.zero_grad()

        _, mean, logvar, max_logvar, min_logvar = model(x)

        inv_var = torch.exp(-logvar)

        mse_loss = (torch.square(mean - y) * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.sum() - 0.01 * min_logvar.sum()

        loss = mse_loss + var_loss + var_bound_loss

        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("Step {}/{} Loss: {:.3f}, MSE: {:.3f}, VAR: {:.3f}, VAR BOUND: {:.3f}"
                  .format(i, steps, loss, mse_loss, var_loss, var_bound_loss))

            if plot:

                f.clear()

                _, mean_plt, logvar_plt, _, _ = model(torch.cat((buffer.obs_buf,
                                                                 buffer.act_buf), dim=1))

                mean_plt = mean_plt.detach().cpu()
                std = torch.exp(0.5*logvar_plt).detach().cpu()

                plot_maze2d_umaze()
                plt.scatter(buffer.obs_buf[train_idx, 0].cpu(),
                            buffer.obs_buf[train_idx, 1].cpu(),
                            marker='x',
                            color='blue')
                plt.scatter(buffer.obs_buf[~train_idx, 0].cpu(),
                            buffer.obs_buf[~train_idx, 1].cpu(),
                            marker='.',
                            color='blue',
                            s=5)

                plt.scatter(mean_plt[0, :, 0],
                            mean_plt[0, :, 1],
                            marker='+',
                            color='red')

                for i in range(len(mean_plt[0])):
                    plt.gca().add_patch(
                        Ellipse((mean_plt[0, i, 0], mean_plt[0, i, 1]),
                                width=std[0, i, 0],
                                height=std[0, i, 1],
                                zorder=-1,
                                color='lightcoral'))

                plt.draw()
                plt.pause(0.001)

    if not plot:
        assert loss.item() < 200


test_probabilistic_model_trains_on_maze2d_umaze(plot=True, steps=100000)
