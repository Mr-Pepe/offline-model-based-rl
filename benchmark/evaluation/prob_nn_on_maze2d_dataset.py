from benchmark.utils.replay_buffer import ReplayBuffer
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

    n_samples = 5000
    train_idx = torch.ones((n_samples), dtype=torch.bool)
    frac = int(0.4*n_samples)
    train_idx[frac:-frac] = 0

    torch.manual_seed(0)

    env = gym.make('maze2d-umaze-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim,
                          act_dim,
                          n_samples,
                          device=device)

    o = env.reset()

    for i in range(n_samples):
        a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        buffer.store(torch.as_tensor(o),
                     torch.as_tensor(a),
                     torch.as_tensor(r),
                     torch.as_tensor(o2),
                     torch.as_tensor(d))

        if d:
            o = env.reset()
        else:
            o = o2

    model = EnvironmentModel(
        obs_dim, act_dim,
        hidden=[200, 200, 200], type='probabilistic', device=device)

    lr = 1e-3
    optim = Adam(model.parameters(), lr=lr)

    loss = torch.tensor(0)

    f = plt.figure()

    x = torch.cat((buffer.obs_buf[train_idx],
                   buffer.act_buf[train_idx]), dim=1)

    next_obs = buffer.obs2_buf[train_idx]
    next_obs += torch.rand_like(next_obs)*0.1-0.05
    y = torch.cat((next_obs,
                   buffer.rew_buf[train_idx].unsqueeze(1)), dim=1)

    x = x.to(device)
    y = y.to(device)

    for i in range(steps):

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

                _, mean_plt, logvar_plt, max_logvar_plt, _ = model(torch.cat((buffer.obs_buf,
                                                                              buffer.act_buf), dim=1))

                mean_plt = mean_plt.detach().cpu()
                std = torch.exp(0.5*logvar_plt).detach().cpu()
                max_std = torch.exp(0.5*max_logvar_plt[0, :].detach().cpu())

                plot_maze2d_umaze()
                plt.scatter(buffer.obs2_buf[train_idx, 0].cpu(),
                            buffer.obs2_buf[train_idx, 1].cpu(),
                            marker='x',
                            color='blue',
                            s=10)
                plt.scatter(buffer.obs2_buf[~train_idx, 0].cpu(),
                            buffer.obs2_buf[~train_idx, 1].cpu(),
                            marker='.',
                            color='blue',
                            s=5)

                plt.scatter(mean_plt[0, :, 0],
                            mean_plt[0, :, 1],
                            marker='+',
                            color='red',
                            s=10)

                for i in range(frac-int(frac*0.2), n_samples-frac+int(frac*0.2), int(n_samples/50)):
                    plt.gca().add_patch(
                        Ellipse((mean_plt[0, i, 0], mean_plt[0, i, 1]),
                                width=std[0, i, 0].abs(),
                                height=std[0, i, 1].abs(),
                                zorder=-1,
                                color='lightcoral'))

                    plt.gca().add_patch(
                        Ellipse((mean_plt[0, i, 0], mean_plt[0, i, 1]),
                                width=max_std[0],
                                height=max_std[1],
                                zorder=-10,
                                color='lightgrey'))

                plt.draw()
                plt.pause(0.001)

    if not plot:
        assert loss.item() < 200


test_probabilistic_model_trains_on_maze2d_umaze(plot=True, steps=500000)
