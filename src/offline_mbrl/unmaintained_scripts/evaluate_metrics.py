import os
import os.path as osp

import numpy as np
from d4rl import get_normalized_score

from offline_mbrl.scripts.metrics import efficiency, final_performance, stability
from offline_mbrl.scripts.plot import get_all_datasets
from offline_mbrl.utils.env_name_from_exp_name import get_env_name_from_experiment_name
from offline_mbrl.utils.envs import HALF_CHEETAH_ENVS, HOPPER_ENVS, WALKER_ENVS
from offline_mbrl.utils.mode_from_exp_name import get_mode_from_experiment_name
from offline_mbrl.utils.modes import (
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY,
    BEHAVIORAL_CLONING,
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY,
    MBPO,
    SAC,
)

if __name__ == "__main__":
    all_exp_dir = ""

    exp_names = [
        name
        for name in os.listdir(all_exp_dir)
        if osp.isdir(osp.join(all_exp_dir, name))
    ]
    exp_names.sort()

    categories = [HALF_CHEETAH_ENVS, HOPPER_ENVS, WALKER_ENVS]

    category_names = ["Halfcheetah", "Hopper", "Walker2D"]

    datasets = [
        "-random-v2",
        "-medium-replay-v2",
        "-medium-v2",
        "-medium-expert-v2",
        "-expert-v2",
    ]

    mode_names = [
        ALEATORIC_PENALTY,
        ALEATORIC_PARTITIONING,
        EPISTEMIC_PENALTY,
        EPISTEMIC_PARTITIONING,
        # BEHAVIORAL_CLONING,
        # SAC,
        # MBPO
    ]

    experiments = {}
    performances = {}

    for exp_name in exp_names:
        exp_dir = osp.join(all_exp_dir, exp_name)
        env_name = get_env_name_from_experiment_name(exp_name)
        mode = get_mode_from_experiment_name(exp_name)

        experiments.update({(env_name, mode): exp_dir})

    for i_category in range(len(categories)):
        for i_dataset, dataset in enumerate(datasets):
            env_name = str.lower(category_names[i_category]) + dataset

            for i_mode, mode in enumerate(mode_names):
                if (env_name, mode) in experiments:
                    data = get_all_datasets(
                        [experiments[(env_name, mode)]], None, None, None
                    )

                    if len(data) != 6:
                        print(env_name, mode)

                    performances[(env_name, mode)] = np.array(
                        [tuple(datum.AverageTestEpRet) for datum in data]
                    ).transpose()

    sorted_keys = sorted(performances.keys(), key=lambda x: x[0] + x[1])

    print("Final performance")
    print("Mean\tStd")
    for (env_name, mode) in sorted_keys:
        perf = list(final_performance(performances[(env_name, mode)]))
        perf[0] = get_normalized_score(env_name, perf[0]) * 100
        perf[1] = get_normalized_score(env_name, perf[1]) * 100

        print(f"{perf[0]:.0f}\t{perf[1]:.0f}\t{env_name} {mode}")

    print("Stability")
    stabilities = []
    for (env_name, mode) in sorted_keys:
        stab = stability(performances[(env_name, mode)])

        stabilities.append((stab, env_name, mode))
        # print(f'{stab:.2f}\t{env_name} {mode}')
    stabilities = sorted(stabilities, key=lambda x: x[0], reverse=True)
    for stab, env_name, mode in stabilities:
        if mode not in [SAC, BEHAVIORAL_CLONING, MBPO]:
            print(f"{stab:.2f}\t{env_name} {mode}")

    print("Efficiency")
    efficiencies = []
    for (env_name, mode) in sorted_keys:
        eff = efficiency(performances[(env_name, mode)])

        efficiencies.append((eff, env_name, mode))
    efficiencies = sorted(efficiencies, key=lambda x: x[0])
    for eff, env_name, mode in efficiencies:
        if mode not in [SAC, BEHAVIORAL_CLONING, MBPO]:
            print(f"{eff:.2f}\t{env_name} {mode}")

    for mode in mode_names:
        elements = []
        for category in category_names:
            for dataset in datasets:
                env_name = category.lower() + dataset

                elements.append(f"{efficiency(performances[(env_name, mode)]):.0f}")

        print(f"{mode}")
        print(" & ".join(elements))
