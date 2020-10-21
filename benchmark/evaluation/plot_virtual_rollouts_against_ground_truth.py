from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.train_environment_model import train_environment_model
from benchmark.utils.virtual_rollouts import generate_virtual_rollout
import gym
import d4rl  # noqa
import torch
from benchmark.models.environment_model import EnvironmentModel
import matplotlib.pyplot as plt
import numpy as np


class GoUpAndRightAgent():
    def __init__(self):
        self.training = False

    def eval(self):
        pass

    def get_action(self, o=0):
        # Always goes up and right
        return torch.tensor([1, 1]).reshape((1, -1))


env = gym.make('maze2d-open-v0')
observation_space = env.observation_space
action_space = env.action_space
buffer, obs_dim, act_dim = load_dataset_from_env(env)
agent = GoUpAndRightAgent()

env.reset()
start_observation = torch.tensor([0.6, 1, 0, 0]).reshape((1, -1))

env.set_state(start_observation[0][:2], start_observation[0][2:])

rollout_length = 50


# Generate a rollout with the actual environment to get the ground truth
rollout = []

for i in range(rollout_length):
    action = agent.get_action()
    o, _, _, _ = env.step(action)
    rollout.append(o)

rollout = np.array(rollout)
plt.plot(rollout[:, 0], rollout[:, 1], color='blue')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EnvironmentModel(obs_dim, act_dim, type='probabilistic')
# model = EnvironmentModel(obs_dim, act_dim, type='deterministic')

model.to(device)

train_environment_model(model, buffer, patience=2, debug=True, lr=1e-3)

virtual_rollout = generate_virtual_rollout(model,
                                           agent,
                                           start_observation,
                                           rollout_length)

virtual_rollout = np.array([list(virtual_rollout[i]['o'][0][:2])
                            for i in range(rollout_length)])

plt.plot(virtual_rollout[:, 0], virtual_rollout[:, 1], color='red')
plt.show()

# Render virtual rollout
# for i in range(rollout_length):
#     env.set_state(virtual_rollout[i]['o'][0][:2],
#                   virtual_rollout[i]['o'][0][2:])
#     env.render()
#     sleep(0.25)
