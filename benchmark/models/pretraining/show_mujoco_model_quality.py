from benchmark.user_config import MODELS_DIR
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM
import torch
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
import matplotlib.pyplot as plt
import os

env_name = HALF_CHEETAH_EXPERT
env = gym.make(env_name)
buffer, obs_dim, act_dim = load_dataset_from_env(env)

model = torch.load(os.path.join(MODELS_DIR, env_name + '-model.pt'))

n_samples = 2000
mopo_pred = None
pepe_pred = None
real_rew = None

env.reset()

random_actions = False

for i_sample in torch.randint(buffer.size, (n_samples,)):
    obs = buffer.obs_buf[i_sample]
    act = torch.as_tensor(env.action_space.sample())

    obs_act = torch.cat((obs, act))

    this_pred = model.get_prediction(obs_act, mode='mopo', pessimism=50)

    if mopo_pred is None:
        mopo_pred = this_pred
    else:
        mopo_pred = torch.cat((mopo_pred, this_pred), dim=0)

    this_pred = model.get_prediction(obs_act, mode='pepe', pessimism=50)

    if pepe_pred is None:
        pepe_pred = this_pred
    else:
        pepe_pred = torch.cat((pepe_pred, this_pred), dim=0)

    if random_actions:
        env.set_state(torch.cat((torch.as_tensor([0]), obs[:8])), obs[8:])
        _, this_pred, _, _ = env.step(act.numpy())
    else:
        this_pred = buffer.rew_buf[i_sample]

    if real_rew is None:
        real_rew = torch.as_tensor(this_pred).unsqueeze(0)
    else:
        real_rew = torch.cat(
            (real_rew, torch.as_tensor(this_pred).unsqueeze(0)))

plt.plot(real_rew, label='Ground truth')
plt.plot(mopo_pred[:, -2].cpu(), label='MOPO')
plt.plot(pepe_pred[:, -2].cpu(), label='Pepe')
plt.legend()
plt.show()
