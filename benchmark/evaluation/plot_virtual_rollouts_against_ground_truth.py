from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.train_environment_model import train_environment_model
from benchmark.utils.virtual_rollouts import generate_virtual_rollout
import gym
import d4rl  # noqa
import torch
from benchmark.models.environment_model import EnvironmentModel
import matplotlib.pyplot as plt
import numpy as np


class GoCircleAgent():
    def __init__(self, length):
        self.training = False
        self.steps = 0
        self.length = length

    def eval(self):
        pass

    def get_action(self, o=0):
        self.steps += 1
        progress = self.steps / self.length
        sections = 4
        if progress < 1/sections:
            return torch.tensor([1, 1]).reshape((1, -1))
        elif progress < 2/sections:
            return torch.tensor([-1, 0]).reshape((1, -1))
        elif progress < 3/sections:
            return torch.tensor([0, -1]).reshape((1, -1))
        else:
            return torch.tensor([0, 0]).reshape((1, -1))

        # if self.steps < 0.5 * self.length:
        #     return torch.tensor([1, 0]).reshape((1, -1))
        # else:
        #     return torch.tensor([0, 1]).reshape((1, -1))


env = gym.make('maze2d-open-v0')
observation_space = env.observation_space
action_space = env.action_space
buffer, obs_dim, act_dim = load_dataset_from_env(env)
rollout_length = 100
agent = GoCircleAgent(rollout_length)

env.reset()
start_observation = torch.tensor([0.6, 1, 0, 0]).reshape((1, -1))

env.set_state(start_observation[0][:2], start_observation[0][2:])


# Generate a rollout with the actual environment to get the ground truth
rollout = []

for i in range(rollout_length):
    action = agent.get_action()
    o, _, _, _ = env.step(action)
    rollout.append(o)

rollout = np.array(rollout)
plt.plot(rollout[:, 0], rollout[:, 1], color='blue', label="Ground truth")


# Train different models and generate rollouts with them
device = 'cuda' if torch.cuda.is_available() else 'cpu'
patience = 20
n_hidden = 256
n_layers = 3
hidden = [n_hidden] * n_layers

models = [EnvironmentModel(obs_dim, act_dim, hidden, type='probabilistic'),
          EnvironmentModel(obs_dim, act_dim, hidden, type='probabilistic',
                           n_networks=3),
          EnvironmentModel(obs_dim, act_dim, hidden, type='deterministic'),
          EnvironmentModel(obs_dim, act_dim, hidden, type='deterministic',
                           n_networks=3)]

names = ['Probabilistic single',
         'Probabilistic ensemble',
         'Deterministic single',
         'Deterministic ensemble']

colors = ['red', 'lightcoral', 'black', 'grey']

for i_model, model in enumerate(models):
    print("Training {}".format(names[i_model]))
    model.to(device)

    train_environment_model(
        model, buffer, patience=patience, debug=True, lr=1e-3)
    model.cpu()

    virtual_rollout = generate_virtual_rollout(model,
                                               agent,
                                               start_observation,
                                               rollout_length,
                                               stop_on_terminal=False)

    virtual_rollout = np.array([list(virtual_rollout[i]['o'][0][:2])
                                for i in range(rollout_length)])

    plt.plot(virtual_rollout[:, 0],
             virtual_rollout[:, 1],
             color=colors[i_model],
             label=names[i_model])

plt.legend()
plt.show()
