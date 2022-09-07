import os
import os.path as osp

import d4rl
import gym
import numpy as np
import torch

from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.env_name_from_exp_name import get_env_name_from_experiment_name
from offline_mbrl.utils.mode_from_exp_name import get_mode_from_experiment_name

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="/home/felipe/Projects/thesis-code/data/offline_exploration",
    )
    args = parser.parse_args()

    all_exp_dir = args.logdir

    exp_names = [
        name
        for name in os.listdir(all_exp_dir)
        if osp.isdir(osp.join(all_exp_dir, name))
    ]
    exp_names.sort()

    for exp_name in exp_names:
        exp_dir = osp.join(all_exp_dir, exp_name)
        env_name = get_env_name_from_experiment_name(exp_name)
        mode = get_mode_from_experiment_name(exp_name)
        env = gym.make(env_name)

        model = torch.load(
            os.path.join(MODELS_DIR, env_name + "-model.pt"), map_location="cpu"
        )

        trial_names = [
            name for name in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, name))
        ]

        # print(f"Found {len(trial_names)} seeds for {env_name} {mode}")

        n_steps = 1000

        obs_act = torch.zeros(
            (
                len(trial_names) * n_steps,
                env.observation_space.shape[0] + env.action_space.shape[0],
            )
        )

        n_terminals = 0
        performances = []

        for i_trial, trial_name in enumerate(trial_names):
            agent = torch.load(
                osp.join(exp_dir, trial_name, "pyt_save", "agent.pt"), "cpu"
            )

            o = env.reset()
            interaction = 0
            performance = 0

            for step in range(n_steps):
                interaction += 1
                a = agent.act(o, deterministic=True).cpu()

                o2, r, d, _ = env.step(a.numpy())
                # o2, r, d, _ = env.step(env.action_space.sample())

                performance += r

                obs_act[i_trial * n_steps + step] = torch.cat((torch.as_tensor(o), a))

                o = o2

                if d and interaction % 1000 != 0:
                    interaction = 0
                    n_terminals += 1
                    o = env.reset()

                    performances.append(performance)
                    performance = 0

        (
            prediction,
            means,
            logvars,
            epistemic_uncertainty,
            aleatoric_uncertainty,
        ) = model.get_prediction(obs_act, debug=True)

        normalized_score = (
            d4rl.get_normalized_score(env_name, np.mean(performance)) * 100
        )

        print(
            f"Uncertainty: {epistemic_uncertainty.mean():.2f}   "
            f"Terminals: {n_terminals}    "
            f"Reward: {normalized_score:.0f}   {env_name} {mode}"
        )
