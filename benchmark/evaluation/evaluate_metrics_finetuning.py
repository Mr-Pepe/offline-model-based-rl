import numpy as np
from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY, BEHAVIORAL_CLONING, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, MBPO, SAC
from benchmark.utils.envs import HALF_CHEETAH_ENVS, HOPPER_ENVS, WALKER_ENVS
import os
import os.path as osp
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.evaluation.plot import get_all_datasets
from orl_metrics.metrics import final_performance
from orl_metrics.metrics import stability
from orl_metrics.metrics import efficiency
from d4rl import get_normalized_score


if __name__ == '__main__':
    all_exp_dir = ''

    categories = [
        HALF_CHEETAH_ENVS,
        HOPPER_ENVS,
        WALKER_ENVS
    ]

    category_names = [
        "Halfcheetah",
        "Hopper",
        "Walker2D"
    ]

    mode_names = [
        ALEATORIC_PENALTY,
        ALEATORIC_PARTITIONING,
        EPISTEMIC_PENALTY,
        EPISTEMIC_PARTITIONING,
    ]

    experiments = dict()
    performances = dict()

    for category in category_names:
        dir = osp.join(all_exp_dir, category.lower())
        exp_dirs = [osp.join(dir, name) for name in os.listdir(
            dir) if osp.isdir(osp.join(dir, name))]
        exp_dirs.sort()

        for exp_dir in exp_dirs:
            env_name = f'{category.lower()}-medium-v2'
            mode = get_mode(exp_dir)

            experiments.update({(env_name, mode): exp_dir})

            data = get_all_datasets([experiments[(env_name, mode)]],
                                    None,
                                    None,
                                    None)

            performances[(env_name, mode)] = np.array(
                [tuple(datum.AverageTestEpRet) for datum in data]).transpose()

    for mode in mode_names:
        elements = []
        for category in category_names:
            env_name = f'{category.lower()}-medium-v2'

            elements.append(
                f'{get_normalized_score(env_name, list(final_performance(performances[(env_name, mode)]))[0])*100:.0f}')

            elements.append(
                f'{get_normalized_score(env_name, list(final_performance(performances[(env_name, mode)]))[1])*100:.0f}')

            elements.append(
                f'{get_normalized_score(env_name, list(final_performance(performances[(env_name, mode)]))[2])*100:.0f}')

            elements.append(
                f'{stability(performances[(env_name, mode)]):.2f}')

            elements.append(
                f'{efficiency(performances[(env_name, mode)]):.0f}')

        print(f'{mode}')
        print(' & '.join(elements))
