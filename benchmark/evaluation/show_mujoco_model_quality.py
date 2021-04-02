from benchmark.utils.modes import ALEATORIC_PENALTY
from matplotlib import cm
import numpy as np
from benchmark.utils.replay_buffer import ReplayBuffer
from benchmark.actors.random_agent import RandomAgent
from benchmark.user_config import MODELS_DIR
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_RANDOM, HOPPER_EXPERT, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_EXPERT_V1, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_V1, HOPPER_RANDOM, WALKER_MEDIUM, WALKER_MEDIUM_EXPERT_V2, WALKER_MEDIUM_REPLAY, WALKER_MEDIUM_REPLAY_V2, WALKER_MEDIUM_v2
import torch
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.actors.sac import SAC
import matplotlib.pyplot as plt
import os

name = "Halfcheetah"
prefix = name.lower() + '-'
version = '-v2'

dataset_names = [
    'random',
    'medium-replay',
    'medium',
    'medium-expert',
]

fig, axes = plt.subplots(2, len(dataset_names))

fig.suptitle(name, fontsize=24)

for i_dataset, dataset_name in enumerate(dataset_names):
    device = 'cuda'
    env_name = prefix + dataset_name + version
    print(env_name)
    env = gym.make(env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env)

    model = torch.load(os.path.join(MODELS_DIR, env_name + '-model.pt'))

    n_samples = 3000
    random_samples = 1000
    i_sample = 0
    batch_size = 256

    idxs = torch.randint(0, buffer.size, (n_samples,))
    obs_buf = buffer.obs_buf[idxs].cuda()
    act_buf = buffer.act_buf[idxs].cuda()
    obs2_buf = buffer.obs2_buf[idxs].cuda()

    model_errors = []
    aleatoric_uncertainties = []
    epistemic_uncertainties = []

    while i_sample < len(obs_buf):
        print(i_sample)
        i_to = min(len(obs_buf), i_sample + batch_size)
        obs_act = torch.cat((obs_buf[i_sample:i_to], act_buf[i_sample:i_to]), dim=1)
        obs2 = obs2_buf[i_sample:i_to]

        prediction, means, logvars, explicit_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, underestimated_reward = model.get_prediction(obs_act, debug=True)

        model_errors.extend((prediction[:, :-2] - obs2).abs().mean(dim=1).cpu().detach().float().tolist())
        aleatoric_uncertainties.extend(aleatoric_uncertainty.tolist())
        epistemic_uncertainties.extend(epistemic_uncertainty.tolist())

        i_sample += batch_size

    if random_samples > 0:
        print("Generating random samples.")
        env.reset()
        agent = RandomAgent(env, device=device)

        for i_obs in range(random_samples):
            obs = obs_buf[i_obs]
            env.set_state(torch.cat((torch.as_tensor([0]), torch.as_tensor(obs[:env.model.nq-1].cpu()))), obs[env.model.nq-1:].cpu())
            act = agent.act()

            obs2, _, _, _ = env.step(act.cpu().detach().numpy())

            obs_act = torch.cat((obs.unsqueeze(0), act), dim=1)

            prediction, means, logvars, explicit_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, underestimated_reward = model.get_prediction(obs_act, debug=True)

            model_errors.extend((prediction[:, :-2].cpu() - obs2).abs().mean(dim=1).cpu().detach().float().tolist())
            aleatoric_uncertainties.extend(aleatoric_uncertainty.tolist())
            epistemic_uncertainties.extend(epistemic_uncertainty.tolist())

    aleatoric_uncertainties = torch.as_tensor(aleatoric_uncertainties)
    aleatoric_uncertainties /= aleatoric_uncertainties.max()
    epistemic_uncertainties = torch.as_tensor(epistemic_uncertainties)
    epistemic_uncertainties /= epistemic_uncertainties.max()
    model_errors = torch.as_tensor(model_errors)
    model_errors /= model_errors.max()

    axes[0, i_dataset].plot([0, 1], [0, 1], color='black')
    axes[0, i_dataset].scatter(model_errors[:n_samples], aleatoric_uncertainties[:n_samples], s=1, alpha=0.3, color='blue')
    axes[0, i_dataset].scatter(model_errors[n_samples:], aleatoric_uncertainties[n_samples:], s=1, alpha=0.3, color='red')
    if i_dataset == 0:
        axes[0, i_dataset].set_ylabel('Aleatoric uncertainty')
    axes[0, i_dataset].set_title(dataset_name)

    axes[1, i_dataset].plot([0, 1], [0, 1], color='black')
    axes[1, i_dataset].scatter(model_errors[:n_samples], epistemic_uncertainties[:n_samples], s=1, alpha=0.3, color='blue')
    axes[1, i_dataset].scatter(model_errors[n_samples:], epistemic_uncertainties[n_samples:], s=1, alpha=0.3, color='red')
    axes[1, i_dataset].set_xlabel('Model error')
    if i_dataset == 0:
        axes[1, i_dataset].set_ylabel('Epistemic uncertainty')

    del obs_buf
    del act_buf
    del obs2_buf

plt.show()
