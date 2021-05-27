import pickle
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
from orl_metrics.metrics import estimation_quality, pessimism

env_names = ['halfcheetah', 'hopper', 'walker2d']

dataset_names = [
    'random-v2',
    'medium-replay-v2',
    'medium-v2',
    'medium-expert-v2',
    'expert-v2',
]

compute = True

results = dict()
if compute:
    for env_title in env_names:

        for i_dataset, dataset_name in enumerate(dataset_names):
            device = 'cuda'
            env_name = env_title + f'-{dataset_name}'
            print(env_name)
            env = gym.make(env_name)

            n_samples = 10000

            buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples)

            model = torch.load(os.path.join(
                MODELS_DIR, env_name + '-model.pt'))

            random_samples = 10000
            i_sample = 0
            batch_size = 1024

            idxs = torch.randint(0, buffer.size, (n_samples,))
            obs_buf = buffer.obs_buf[idxs].cuda()
            act_buf = buffer.act_buf[idxs].cuda()
            obs2_buf = buffer.obs2_buf[idxs].cuda()

            model_errors = []
            aleatoric_uncertainties = []
            epistemic_uncertainties = []

            while i_sample < len(obs_buf):
                print(f'ID: {i_sample}/{len(obs_buf)}', end='\r')
                i_to = min(len(obs_buf), i_sample + batch_size)
                obs_act = torch.cat(
                    (obs_buf[i_sample:i_to], act_buf[i_sample:i_to]), dim=1)
                obs2 = obs2_buf[i_sample:i_to]

                prediction, means, logvars, explicit_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, underestimated_reward = model.get_prediction(
                    obs_act, debug=True)

                model_errors.extend(
                    (prediction[:, :-2] - obs2).abs().mean(dim=1).cpu().detach().float().tolist())
                aleatoric_uncertainties.extend(aleatoric_uncertainty.tolist())
                epistemic_uncertainties.extend(epistemic_uncertainty.tolist())

                i_sample += batch_size

            aleatoric_uncertainties = torch.as_tensor(aleatoric_uncertainties)
            epistemic_uncertainties = torch.as_tensor(epistemic_uncertainties)
            model_errors = torch.as_tensor(model_errors)

            al_corr = estimation_quality(model_errors, aleatoric_uncertainties)
            ep_corr = estimation_quality(model_errors, epistemic_uncertainties)
            al_over = pessimism(model_errors, aleatoric_uncertainties)
            ep_over = pessimism(model_errors, epistemic_uncertainties)
            print(
                f'ID  (al/ep) (al/ep): {al_corr:.2f}/{ep_corr:.2f}  {al_over:.2f}/{ep_over:.2f}')

            results[(env_name, dataset_name, 'ID')] = (
                al_corr, ep_corr, al_over, ep_over)

            model_errors = []
            aleatoric_uncertainties = []
            epistemic_uncertainties = []

            if random_samples > 0:
                env.reset()
                agent = RandomAgent(env, device=device)

                for i_obs in range(random_samples):
                    print(f'OOD: {i_obs}/{random_samples}', end='\r')
                    obs = obs_buf[i_obs]
                    env.set_state(torch.cat((torch.as_tensor([0]), torch.as_tensor(
                        obs[:env.model.nq-1].cpu()))), obs[env.model.nq-1:].cpu())
                    act = agent.act()

                    obs2, _, _, _ = env.step(act.cpu().detach().numpy())

                    obs_act = torch.cat((obs.unsqueeze(0), act), dim=1)

                    prediction, means, logvars, explicit_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, underestimated_reward = model.get_prediction(
                        obs_act, debug=True)

                    model_errors.extend(
                        (prediction[:, :-2].cpu() - obs2).abs().mean(dim=1).cpu().detach().float().tolist())
                    aleatoric_uncertainties.extend(
                        aleatoric_uncertainty.tolist())
                    epistemic_uncertainties.extend(
                        epistemic_uncertainty.tolist())

            aleatoric_uncertainties = torch.as_tensor(aleatoric_uncertainties)
            epistemic_uncertainties = torch.as_tensor(epistemic_uncertainties)
            model_errors = torch.as_tensor(model_errors)

            al_corr = estimation_quality(model_errors, aleatoric_uncertainties)
            ep_corr = estimation_quality(model_errors, epistemic_uncertainties)
            al_over = pessimism(model_errors, aleatoric_uncertainties)
            ep_over = pessimism(model_errors, epistemic_uncertainties)
            print(
                f'OOD  (al/ep) (al/ep): {al_corr:.2f}/{ep_corr:.2f}  {al_over:.2f}/{ep_over:.2f}')

            results[(env_name, dataset_name, 'OOD')] = (
                al_corr, ep_corr, al_over, ep_over)
            del env
            del buffer
            del model

    with open('model-quality.p', 'wb') as f:
        pickle.dump(results, f)

with open('model-quality.p', 'rb') as f:
    results = pickle.load(f)

strings = [[], [], [], [], [], [], [], []]

for env_title in env_names:
    for i_dataset, dataset_name in enumerate(dataset_names):
        device = 'cuda'
        env_name = env_title + f'-{dataset_name}'

        result_id = results[(env_name, dataset_name, 'ID')]
        result_ood = results[(env_name, dataset_name, 'OOD')]

        # MSE ID Ep
        strings[0].append(f'{result_id[1].item():.2f}')
        # MSE ID Al
        strings[1].append(f'{result_id[0].item():.2f}')
        # MSE OOD Ep
        strings[2].append(f'{result_ood[1].item():.2f}')
        # MSE OOD Al
        strings[3].append(f'{result_ood[0].item():.2f}')
        # PS ID Ep
        strings[4].append(f'{result_id[3].item():.2f}')
        # PS ID Al
        strings[5].append(f'{result_id[2].item():.2f}')
        # PS OOD Ep
        strings[6].append(f'{result_ood[3].item():.2f}')
        # PS OOD Al
        strings[7].append(f'{result_ood[2].item():.2f}')

for x in [' & '.join(x) for x in strings]:
    print(x)
# latex = ' \\\\ '.join([' & '.join(x) for x in strings])
# print(latex)
