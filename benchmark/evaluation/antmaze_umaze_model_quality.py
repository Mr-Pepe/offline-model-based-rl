from benchmark.utils.replay_buffer import ReplayBuffer
import torch
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.mazes import plot_antmaze_umaze
import matplotlib.pyplot as plt
import time

env = gym.make('antmaze-umaze-v0')
original_buffer, obs_dim, act_dim = load_dataset_from_env(env, 100)
real_buffer = ReplayBuffer(obs_dim, act_dim, original_buffer.size)
virtual_buffer = ReplayBuffer(obs_dim, act_dim, original_buffer.size)

agent = None
# torch.load(
#     '/home/felipe/Projects/thesis-code/data/antmaze_umaze_mopo/antmaze_umaze_mopo_s0/pyt_save/agent.pt')
# agent.cpu()
model = torch.load(
    '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/model.pt')

obs = original_buffer.obs_buf[0]

# What would happen in the virtual environment
for step in range(original_buffer.size):
    if not agent:
        # act = original_buffer.act_buf[step]
        act = torch.zeros_like(original_buffer.act_buf[0])
    else:
        act = agent.act(obs)

    pred = model.get_prediction(torch.cat((obs, act)).unsqueeze(0))

    virtual_buffer.store(obs, act, pred[:, -2], pred[:, :-2], 0)

    obs = pred[0, :-2]

obs = original_buffer.obs_buf[0]
env.reset()
# See what the agents actions would have led to in the real environment
# if agent:
for step in range(500):
    # act = agent.act(obs, False)
    act = torch.zeros_like(original_buffer.act_buf[0])
    # env.set_state(obs[:15].view(-1), obs[15:].view(-1))
    o2, r, d, _ = env.step(act.cpu().numpy())
    # env.render()
    real_buffer.store(torch.as_tensor(obs),
                      torch.as_tensor(act),
                      torch.as_tensor(r),
                      torch.as_tensor(o2),
                      torch.as_tensor(d))

    obs = torch.as_tensor(o2)

for step in range(original_buffer.size):
    # obs = original_buffer.obs_buf[step]
    # env.set_state(obs[:15].view(-1), obs[15:].view(-1))
    # env.render()
    # time.sleep(0.1)

    obs = real_buffer.obs_buf[step]
    env.set_state(obs[:15].view(-1), obs[15:].view(-1))
    env.render()
    time.sleep(0.1)

    obs = virtual_buffer.obs_buf[step]
    env.set_state(obs[:15].view(-1), obs[15:].view(-1))
    env.render()
    time.sleep(0.1)

# plot_antmaze_umaze(buffer=original_buffer)
# plt.scatter(virtual_buffer.obs2_buf[:, 0], virtual_buffer.obs2_buf[:, 1])
# plt.show()

pass
