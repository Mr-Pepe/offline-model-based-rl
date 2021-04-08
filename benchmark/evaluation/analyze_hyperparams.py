from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, PENALTY_MODES
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.utils.mode_from_exp_name import get_mode
import os
import os.path as osp
import csv
import d4rl
import json
import matplotlib.pyplot as plt


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
            epoch = 5
            if len(trial_log) >= epoch:
                final = 100 * d4rl.get_normalized_score(
                    env_name, float(trial_log[epoch-1]['avg_test_return']))
                returns = [100 * d4rl.get_normalized_score(
                    env_name, float(trial_log[i]['test_return'])) for i in range(epoch)]

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

    name = "Halfcheetah"
    prefix = name.lower() + '-'
    version = '-v2'

    dataset_names = [
        'random',
        'medium-replay',
        'medium',
        'medium-expert',
        'expert'
    ]

    modes = [
        ALEATORIC_PARTITIONING,
        EPISTEMIC_PARTITIONING,
        ALEATORIC_PENALTY,
        EPISTEMIC_PENALTY,
    ]

    f, axes = plt.subplots(len(modes), len(dataset_names))
    f.suptitle(name, fontsize=18)

    for i_dataset, dataset_name in enumerate(dataset_names):
        for i_mode, mode in enumerate(modes):
            env_name = prefix + dataset_name + version

            if (env_name, mode) in experiments:
                ax = axes[i_mode, i_dataset]
                trials = experiments[(env_name, mode)]

                trials = zip([trials[i]['rollout_length'] for i in range(len(trials))],
                             [trials[i]['pessimism']
                                 for i in range(len(trials))],
                             [trials[i]['final_return'] for i in range(len(trials))])

                trials = sorted(trials, key=lambda x: x[2], reverse=True)
                x = [trial[0] for trial in trials]
                y = [trial[1] for trial in trials]
                y = [e/max(y) for e in y]
                z = [trial[2] for trial in trials]

                ax.scatter(x, y, s=8)
                ax.scatter(
                    x[:8], y[:8], marker='o',  s=40, facecolors='none', edgecolors='red')
                ax.set_xlim([-1, 21])
                ax.set_ylim([-0.05, 1.05])

                if i_dataset == 0:
                    if mode in PENALTY_MODES:
                        ax.set_ylabel('Penalty', fontsize=10)
                    else:
                        ax.set_ylabel('Threshold', fontsize=10)
                    ax.set_yticks([0, 1])
                else:
                    ax.set_ylabel(None)
                    ax.set_yticks([])

                if i_mode == len(modes) - 1:
                    ax.set_xlabel('Rollout\nlength', fontsize=10)
                    ax.set_xticks([0, 20])
                else:
                    ax.set_xlabel(None)
                    ax.set_xticks([])

    pad = 5

    for ax, col in zip(axes[0], [name.replace('-', '\n') for name in dataset_names]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=12, ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], [name.replace('-', '\n') for name in modes]):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=12, ha='right', va='center', rotation=0)

    f.subplots_adjust(top=0.81,
                      bottom=0.135,
                      left=0.245,
                      right=0.9,
                      hspace=0.105,
                      wspace=0.1)
    plt.show()
