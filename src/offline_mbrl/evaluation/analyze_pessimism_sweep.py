import csv
import json
import os
import os.path as osp

import d4rl  # noqa
import numpy as np
from matplotlib import pyplot as plt

from offline_mbrl.utils.env_name_from_exp_name import get_env_name_from_experiment_name
from offline_mbrl.utils.mode_from_exp_name import get_mode_from_experiment_name
from offline_mbrl.utils.modes import (
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY,
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY,
    PARTITIONING_MODES,
    PENALTY_MODES,
)


def main():
    exp_dirs = []

    al_max = 0.015116
    ep_max = 1.014396

    f, axes = plt.subplots(1, 4, figsize=(8, 3))

    for i_exp, exp_dir in enumerate(exp_dirs):
        ax = axes[i_exp]

        env_name = get_env_name_from_experiment_name(exp_dir)
        mode = get_mode_from_experiment_name(exp_dir)

        trial_names = [
            name for name in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, name))
        ]

        trial_performances = []
        pessimisms = []

        for trial_name in trial_names:
            with open(
                osp.join(exp_dir, trial_name, "progress.csv"), encoding="utf-8"
            ) as f:
                trial_log = list(csv.DictReader(f, delimiter=","))

            with open(
                osp.join(exp_dir, trial_name, "params.json"), encoding="utf-8"
            ) as f:
                params = json.load(f)

            returns = [
                100
                * d4rl.get_normalized_score(
                    env_name, float(trial_log[i]["test_return"])
                )
                for i in range(len(trial_log))
            ]

            if mode in PENALTY_MODES:
                pessimisms.append(params["model_pessimism"])
            else:
                pessimisms.append(params["ood_threshold"])

            trial_performances.append(returns[-1])

        pessimisms, trial_performances = zip(
            *sorted(zip(pessimisms, trial_performances), key=lambda x: x[0])
        )

        if mode == ALEATORIC_PENALTY:
            pessimisms = np.array(pessimisms) * al_max
            max_pessimism = 50
            kernel_size = 50
        elif mode == EPISTEMIC_PENALTY:
            pessimisms = np.array(pessimisms) * ep_max
            max_pessimism = 100
            kernel_size = 60
        elif mode == ALEATORIC_PARTITIONING:
            pessimisms = np.array(pessimisms) / al_max
            max_pessimism = 1
            kernel_size = 30
        elif mode == EPISTEMIC_PARTITIONING:
            pessimisms = np.array(pessimisms) / ep_max
            max_pessimism = 1
            kernel_size = 30

        kernel = np.ones(kernel_size) / kernel_size
        trial_performances = np.convolve(trial_performances, kernel, mode="same")

        ax.plot(
            pessimisms[pessimisms < max_pessimism],
            trial_performances[pessimisms < max_pessimism],
        )
        ax.set_ylim([0, 120])
        ax.set_xlim([-0.1 * max_pessimism, max_pessimism * 1.1])

        if mode in PARTITIONING_MODES:
            ax.set_xlabel("Partitioning\nthreshold")
        elif mode in PENALTY_MODES:
            ax.set_xlabel("Reward penalty\ncoefficient")

        ax.grid(b=True, alpha=0.5, linestyle="--")
        ax.set_title(mode.replace("-", "\n"))

    axes[0].set_ylabel("Performance")
    plt.subplots_adjust(
        top=0.87, bottom=0.19, left=0.075, right=0.995, hspace=0.2, wspace=0.34
    )

    plt.show()


if __name__ == "__main__":
    main()
