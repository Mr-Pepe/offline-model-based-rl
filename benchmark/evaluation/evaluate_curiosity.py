from benchmark.utils.modes import PENALTY_MODES
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.print_warning import print_warning
from benchmark.user_config import MODELS_DIR
import os
import os.path as osp
import csv
import torch
import d4rl
import numpy as np
import gym


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str, default='/home/felipe/Projects/thesis-code/data/offline_exploration')
    args = parser.parse_args()

    all_exp_dir = args.logdir

    exp_names = [name for name in os.listdir(
        all_exp_dir) if osp.isdir(osp.join(all_exp_dir, name))]
    exp_names.sort()

    for exp_name in exp_names:
        exp_dir = osp.join(all_exp_dir, exp_name)
        env_name = get_env_name(exp_name)
        mode = get_mode(exp_name)
        env = gym.make(env_name)

        model = torch.load(os.path.join(
            MODELS_DIR, env_name + '-model.pt'))

        trial_names = [name for name in os.listdir(
            exp_dir) if osp.isdir(osp.join(exp_dir, name))]

        print(f"Found {len(trial_names)} seeds for {env_name} {mode}")

        n_steps = 1000

        obs_act = torch.zeros(
            (len(trial_names) * n_steps, env.observation_space.shape[0] + env.action_space.shape[0]))

        for i_trial, trial_name in enumerate(trial_names):
            agent = torch.load(
                osp.join(exp_dir, trial_name, 'pyt_save', 'agent.pt'))

            n_terminals = 0

            o = env.reset()

            for step in range(n_steps):
                a = agent.act(o, deterministic=True).cpu()

                o2, r, d, _ = env.step(a.numpy())
                # o2, r, d, _ = env.step(env.action_space.sample())

                obs_act[i_trial * n_steps +
                        step] = torch.cat((torch.as_tensor(o), a))

                o = o2

                if d and (step + 1) % 1000 != 0:
                    n_terminals += 1
                    o = env.reset()

        prediction, means, logvars, explicit_uncertainties, epistemic_uncertainty, aleatoric_uncertainty, underestimated_reward = model.get_prediction(
            obs_act, debug=True)

        print(
            f"Mean epistemic uncertainty: {epistemic_uncertainty.mean()}   Number of terminals: {n_terminals}")
