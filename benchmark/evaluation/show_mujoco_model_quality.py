from benchmark.utils.modes import ALEATORIC_PENALTY
from matplotlib import cm
import numpy as np
from benchmark.utils.replay_buffer import ReplayBuffer
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


env_name = WALKER_MEDIUM_EXPERT_V2
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
pessimism = 1
mode = ALEATORIC_PENALTY

for i_rollout in range(n_rollouts):
    (rollouts, info) = generate_virtual_rollouts(model,
                                                 agent,
                                                 buffer, steps=steps,
                                                 n_rollouts=1,
                                                 pessimism=10,
                                                 ood_threshold=0.5,
                                                 mode=mode,
                                                 random_action=True)

    real_rew = None
    means = None
    logvars = None
    explicit_uncertainties = None
    epistemic_uncertainties = None
    aleatoric_uncertainties = None
    underestimated_rewards = None

    for i in range(rollouts['rew'].shape[0]):
        obs = rollouts['obs'][i].cpu()
        act = rollouts['act'][i].cpu()

        obs_act = torch.cat((obs, act))

        predictions, this_means, this_logvars, this_explicit_uncertainty, \
            this_epistemic_uncertainty, this_aleatoric_uncertainty, this_underestimated_reward = model.get_prediction(
                obs_act, mode=mode, pessimism=pessimism, debug=True)

        env = gym.make(env_name)
        env.reset()
        env.set_state(
            torch.cat((torch.as_tensor([0]), obs[:env.model.nq-1])), obs[env.model.nq-1:])
        # env.render()
        _, r, _, _ = env.step(act.numpy())

        if explicit_uncertainties is None:
            means = this_means
            logvars = this_logvars
            explicit_uncertainties = this_explicit_uncertainty
            epistemic_uncertainties = this_epistemic_uncertainty
            aleatoric_uncertainties = this_aleatoric_uncertainty
            underestimated_rewards = this_underestimated_reward
            real_rew = torch.as_tensor(r).unsqueeze(0)
        else:
            means = torch.cat((means, this_means), dim=1)
            logvars = torch.cat((logvars, this_logvars), dim=1)
            explicit_uncertainties = torch.cat(
                (explicit_uncertainties, this_explicit_uncertainty), dim=1)
            epistemic_uncertainties = torch.cat(
                (epistemic_uncertainties, this_epistemic_uncertainty))
            aleatoric_uncertainties = torch.cat(
                (aleatoric_uncertainties, this_aleatoric_uncertainty))
            underestimated_rewards = torch.cat(
                (underestimated_rewards, this_underestimated_reward))
            real_rew = torch.cat(
                (real_rew, torch.as_tensor(r).unsqueeze(0)))

    print("Reward overestimation percentage for rollout length {}: {:.2f}%".format(
        steps,
        ((real_rew - underestimated_rewards.cpu()) < 0).sum().float() / real_rew.numel() * 100))

    r_means = means[:, :, -1].detach().cpu()
    r_logvars = logvars[:, :, -1].detach().cpu()

    f, axes = plt.subplots(4, 1)

    for i in range(r_means.shape[0]):
        axes[0].fill_between(range(r_means.shape[-1]), r_means[i]+torch.exp(
            r_logvars[i]), r_means[i]-torch.exp(r_logvars[i]), alpha=0.5)
    axes[0].plot(real_rew, label='Ground truth')
    axes[0].plot(underestimated_rewards.cpu(), label='Underestimated reward')
    axes[0].legend(fontsize=12)
    axes[0].set_ylim([buffer.rew_buf.min().cpu(), buffer.rew_buf.max().cpu()])
    axes[0].set_ylabel("Reward", fontsize=12)
    axes[0].set_xlabel("Steps", fontsize=12)

    color = cm.rainbow(np.linspace(0, 1, model.n_networks))
    for i_network, c in zip(range(model.n_networks), color):
        axes[1].plot(
            explicit_uncertainties[i_network].detach().cpu().numpy(), color=c)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].plot(explicit_uncertainties.mean(dim=0).detach().cpu(), color='black')
    axes[1].set_ylabel("Explicit", fontsize=12)

    axes[2].plot(epistemic_uncertainties.detach().cpu())
    axes[2].set_ylabel("Epistemic", fontsize=12)

    axes[3].plot(aleatoric_uncertainties.detach().cpu())
    axes[3].set_ylabel("Aleatoric", fontsize=12)

    axes[3].set_xlabel("Steps", fontsize=12)

    plt.tight_layout()
    plt.show()
