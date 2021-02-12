from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY
from matplotlib import cm
import numpy as np
from benchmark.utils.replay_buffer import ReplayBuffer
from benchmark.user_config import MODELS_DIR
from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_RANDOM, HOPPER_EXPERT, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_EXPERT_V1, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_V1, HOPPER_RANDOM, WALKER_MEDIUM, WALKER_MEDIUM_REPLAY
import torch
import gym
import d4rl  # noqa
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.actors.sac import SAC
import matplotlib.pyplot as plt
import os


env_name = HOPPER_MEDIUM
env = gym.make(env_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(os.path.join(MODELS_DIR, env_name +
                                '-model.pt'), map_location=device)

buffer, obs_dim, act_dim = load_dataset_from_env(
    env, buffer_device=next(model.parameters()).device)

batch_size = 4096

explicit_uncertainties = None
epistemic_uncertainties = None
aleatoric_uncertainties = None

for i in range(0, buffer.size, batch_size):
    obs = buffer.obs_buf[i:i+batch_size]
    act = buffer.act_buf[i:i+batch_size]
    obs_act = torch.cat((obs, act), dim=1)

    predictions, this_means, this_logvars, this_explicit_uncertainty, \
        this_epistemic_uncertainty, this_aleatoric_uncertainty = model.get_prediction(
            obs_act, mode=ALEATORIC_PARTITIONING, debug=True)

    if explicit_uncertainties is None:
        explicit_uncertainties = this_explicit_uncertainty
        epistemic_uncertainties = this_epistemic_uncertainty
        aleatoric_uncertainties = this_aleatoric_uncertainty
    else:
        explicit_uncertainties = torch.cat(
            (explicit_uncertainties, this_explicit_uncertainty), dim=1)
        epistemic_uncertainties = torch.cat(
            (epistemic_uncertainties, this_epistemic_uncertainty))
        aleatoric_uncertainties = torch.cat(
            (aleatoric_uncertainties, this_aleatoric_uncertainty))

explicit_uncertainties = explicit_uncertainties.mean(dim=0)

print("\nEnvironment: {}\n".format(env_name))
print("Reward span: {}".format(buffer.rew_buf.max() - buffer.rew_buf.min()))
print("Aleatoric max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(aleatoric_uncertainties.max(), aleatoric_uncertainties.mean(), aleatoric_uncertainties.std()))
print("Epistemic max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(epistemic_uncertainties.max(), epistemic_uncertainties.mean(), epistemic_uncertainties.std()))
print("Explicit  max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(explicit_uncertainties.max(), explicit_uncertainties.mean(), explicit_uncertainties.std()))
