from matplotlib import cm
import numpy as np
from benchmark.utils.replay_buffer import ReplayBuffer
from benchmark.user_config import MODELS_DIR
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_RANDOM, HOPPER_EXPERT, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_V1, HOPPER_RANDOM, WALKER_MEDIUM, WALKER_MEDIUM_REPLAY
import torch
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.actors.sac import SAC
import matplotlib.pyplot as plt
import os


env_name = WALKER_MEDIUM_REPLAY
env = gym.make(env_name)

model = torch.load(os.path.join(MODELS_DIR, env_name +
                                '-model.pt'), map_location="cpu")
agent = SAC(env.observation_space, env.action_space,
            device=next(model.parameters()).device)

buffer, obs_dim, act_dim = load_dataset_from_env(
    env, buffer_device=next(model.parameters()).device)

virtual_buffer = ReplayBuffer(obs_dim, act_dim, size=100000)

steps = 200
n_rollouts = 50
mode = 'mopo'

for i_rollout in range(n_rollouts):
    (rollouts, info) = generate_virtual_rollouts(model,
                                                 agent,
                                                 buffer, steps=steps,
                                                 n_rollouts=1,
                                                 pessimism=10,
                                                 ood_threshold=0.5,
                                                 mode=mode,
                                                 random_action=True)

    env.reset()
    real_rew = None
    r_means = None
    r_logvars = None
    uncertainties = None

    for i in range(rollouts['rew'].shape[0]):
        obs = rollouts['obs'][i].cpu()
        act = rollouts['act'][i].cpu()

        obs_act = torch.cat((obs, act))

        predictions, means, logvars, _, _, uncertainty = model(obs_act)

        this_r_means = means[:, :, -1]
        this_r_logvars = logvars[:, :, -1]

        if uncertainties is None:
            uncertainties = uncertainty
        else:
            uncertainties = torch.cat((uncertainties, uncertainty), dim=1)

        if r_means is None:
            r_means = this_r_means
        else:
            r_means = torch.cat((r_means, this_r_means), dim=1)

        if r_logvars is None:
            r_logvars = this_r_logvars
        else:
            r_logvars = torch.cat((r_logvars, this_r_logvars), dim=1)

        env.set_state(
            torch.cat((torch.as_tensor([0]), obs[:env.model.nq-1])), obs[env.model.nq-1:])
        _, r, _, _ = env.step(act.numpy())

        if real_rew is None:
            real_rew = torch.as_tensor(r).unsqueeze(0)
        else:
            real_rew = torch.cat(
                (real_rew, torch.as_tensor(r).unsqueeze(0)))

    print("Reward overestimation percentage for rollout length {}: {:.2f}%".format(
        steps,
        ((real_rew - rollouts['rew'].cpu()) < 0).sum().float() / real_rew.numel() * 100))

    r_means = r_means.detach().cpu()
    r_logvars = r_logvars.detach().cpu()

    f, axes = plt.subplots(2, 1)

    for i in range(r_means.shape[0]):
        axes[0].fill_between(range(r_means.shape[-1]), r_means[i]+torch.exp(
            r_logvars[i]), r_means[i]-torch.exp(r_logvars[i]), alpha=0.5)
    axes[0].plot(real_rew, label='Ground truth')
    axes[0].plot(rollouts['rew'].cpu(), label=mode)
    axes[0].legend(fontsize=12)
    axes[0].set_ylim([buffer.rew_buf.min().cpu(), buffer.rew_buf.max().cpu()])
    axes[0].set_ylabel("Reward", fontsize=12)
    axes[0].set_xlabel("Steps", fontsize=12)

    color = cm.rainbow(np.linspace(0, 1, model.n_networks))
    for i_network, c in zip(range(model.n_networks), color):
        axes[1].plot(uncertainties[i_network].detach().cpu().numpy(), color=c)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].plot(uncertainties.mean(dim=0).detach().cpu().numpy(), color='black')
    plt.tight_layout()
    plt.show()
