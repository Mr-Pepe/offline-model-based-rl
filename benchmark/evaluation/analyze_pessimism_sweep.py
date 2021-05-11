from matplotlib import pyplot as plt
from benchmark.utils.modes import PARTITIONING_MODES, PENALTY_MODES
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.env_name_from_exp_name import get_env_name
import os
import os.path as osp
import csv
import json
import d4rl  # noqa
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str, default='/home/felipe/ray_results12/sweephopper-expert-v2-epistemic-penalty')
    args = parser.parse_args()

    env_name = get_env_name(args.logdir)
    mode = get_mode(args.logdir)

    trial_names = [name for name in os.listdir(
        args.logdir) if osp.isdir(osp.join(args.logdir, name))]

    trial_performances = []
    pessimisms = []

    for trial_name in trial_names:
        with open(osp.join(args.logdir, trial_name, 'progress.csv'), 'r') as f:
            trial_log = list(csv.DictReader(f, delimiter=','))

        with open(osp.join(args.logdir, trial_name, 'params.json'), 'r') as f:
            params = json.load(f)

        final = 100 * d4rl.get_normalized_score(
            env_name, float(trial_log[-1]['avg_test_return']))
        returns = [100 * d4rl.get_normalized_score(
            env_name, float(trial_log[i]['test_return'])) for i in range(len(trial_log))]

        if mode in PENALTY_MODES:
            pessimisms.append(params['model_pessimism'])
        else:
            pessimisms.append(params['ood_threshold'])

        trial_performances.append(final)

    pessimisms, trial_performances = zip(
        *sorted(zip(pessimisms, trial_performances), key=lambda x: x[0]))

    kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    trial_performances = np.convolve(trial_performances, kernel, mode='same')

    plt.plot(pessimisms, trial_performances)
    plt.ylabel('Performance', fontsize=20)
    plt.ylim([0, 100])

    if mode in PARTITIONING_MODES:
        plt.xlabel("Partitioning threshold", fontsize=20)
    elif mode in PENALTY_MODES:
        plt.xlabel("Reward penalty coefficient", fontsize=20)

    plt.grid(b=True, alpha=0.5, linestyle="--")
    plt.subplots_adjust(top=0.935,
                        bottom=0.14,
                        left=0.135,
                        right=0.97,
                        hspace=0.2,
                        wspace=0.2)

    plt.show()
