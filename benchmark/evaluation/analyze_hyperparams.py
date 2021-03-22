from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, PENALTY_MODES
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.print_warning import print_warning
import os
import os.path as osp
import csv
import torch
import d4rl
import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import DrawingArea


def plot_hyperparameters(env_name, mode, trials):
    x = [trials[i]['rollout_length'] for i in range(len(trials))]
    y = [trials[i]['pessimism'] for i in range(len(trials))]
    z = [trials[i]['final_return'] for i in range(len(trials))]

    xlabel = 'Rollout length'

    if mode in PENALTY_MODES:
        ylabel = 'Penalty coefficient'
    else:
        ylabel = 'OOD threshold'

    fig, axes = plt.subplots(1, 2)

    sc = axes[0].scatter(x, y, c=z)
    axes[0].set_title(env_name + ' ' + mode)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    # axes[0].colorbar()

    def hover(event):
        if event.inaxes == axes[0]:
            cont, ind = sc.contains(event)
            if cont:
                idx = int(ind["ind"][0])
                axes[1].clear()
                axes[1].set_title("{:.2f}  Rollouts: {}  Pessimism: {:.5f}".format(trials[idx]['final_return'],
                                                                                   trials[idx]['rollout_length'],
                                                                                   trials[idx]['pessimism']))
                axes[1].plot(trials[idx]['returns'])
                axes[1].set_ylim([-5, 105])
                fig.canvas.draw_idle()
            else:
                pass

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str, default='/home/felipe/ray_results')
    args = parser.parse_args()

    all_exp_dir = args.logdir

    exp_names = [name for name in os.listdir(
        all_exp_dir) if osp.isdir(osp.join(all_exp_dir, name))]
    exp_names.sort()

    experiments = dict()

    for exp_name in exp_names:
        exp_dir = osp.join(all_exp_dir, exp_name)
        env_name = get_env_name(exp_name)
        mode = get_mode(exp_name)

        trial_names = [name for name in os.listdir(
            exp_dir) if osp.isdir(osp.join(exp_dir, name))]

        trial_performances = []

        for trial_name in trial_names:
            with open(osp.join(exp_dir, trial_name, 'progress.csv'), 'r') as f:
                trial_log = list(csv.DictReader(f, delimiter=','))

            with open(osp.join(exp_dir, trial_name, 'params.json'), 'r') as f:
                params = json.load(f)

            # Only consider trials that ran for 10 iterations
            if len(trial_log) == 10:
                final = 100 * d4rl.get_normalized_score(
                    env_name, float(trial_log[-1]['avg_test_return']))
                returns = [100 * d4rl.get_normalized_score(
                    env_name, float(trial_log[i]['test_return'])) for i in range(len(trial_log))]

                if mode in PENALTY_MODES:
                    pessimism = params['model_pessimism']
                else:
                    pessimism = params['ood_threshold']

                if (env_name, mode) not in experiments:
                    experiments[(env_name, mode)] = []

                experiments[(env_name, mode)].append(dict(pessimism=pessimism,
                                                          rollout_length=params['max_rollout_length'],
                                                          final_return=final,
                                                          returns=returns))

    prefix = 'walker2d-'
    version = '-v2'

    dataset_names = [
        # 'random',
        # 'medium-replay',
        'medium',
        # 'medium-expert',
        # 'expert',
    ]

    modes = [
        ALEATORIC_PARTITIONING,
        EPISTEMIC_PARTITIONING,
        ALEATORIC_PENALTY,
        EPISTEMIC_PENALTY,
    ]

    for dataset_name in dataset_names:
        for mode in modes:
            env_name = prefix + dataset_name + version

            if (env_name, mode) in experiments:
                plot_hyperparameters(
                    env_name, mode, experiments[(env_name, mode)])
