# Based on https://spinningup.openai.com
from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY
from benchmark.utils.envs import HALF_CHEETAH_ENVS, HALF_CHEETAH_MEDIUM_V2, HOPPER_ENVS, HOPPER_MEDIUM_EXPERT_V2, HOPPER_MEDIUM_REPLAY_V2, HOPPER_MEDIUM_V2, WALKER_ENVS, WALKER_MEDIUM_EXPERT_V2, WALKER_MEDIUM_REPLAY_V2, WALKER_MEDIUM_v2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.utils.str2bool import str2bool
from d4rl import get_normalized_score

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1",
              smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    plt.legend(loc='best', ncol=1, handlelength=1,
               borderaxespad=0., prop={'size': 13})

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific
        # notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except Exception:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except Exception:
                print('Could not read from %s' %
                      os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data \
                else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns),
                            'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            def fulldir(x): return osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(
            not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None,
               estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    # choose what to show on main curve: mean? max? min?
    estimator = getattr(np, estimator)
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition,
                  smooth=smooth, estimator=estimator)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', default=[
                        '/home/felipe/Projects/thesis-code/data/experiments'], nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Epoch')
    parser.add_argument(
        '--value', '-y', default=['AverageTestEpRet'], nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--final_eval', type=str2bool, default=True)
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.

    """

    if not args.final_eval:
        make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
                   smooth=args.smooth, select=args.select, exclude=args.exclude,
                   estimator=args.est)

    else:
        all_exp_dir = args.logdir[0]

        exp_names = [name for name in os.listdir(
            all_exp_dir) if osp.isdir(osp.join(all_exp_dir, name))]
        exp_names.sort()

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

        datasets = [
            '-medium-replay-v2',
            '-medium-v2',
            '-medium-expert-v2'
        ]

        mode_names = [
            ALEATORIC_PENALTY,
            ALEATORIC_PARTITIONING,
            EPISTEMIC_PENALTY,
            EPISTEMIC_PARTITIONING
        ]

        experiments = dict()

        for exp_name in exp_names:
            exp_dir = osp.join(all_exp_dir, exp_name)
            env_name = get_env_name(exp_name)
            mode = get_mode(exp_name)

            experiments.update({(env_name, mode): exp_dir})

        f, axes = plt.subplots(len(categories), len(datasets))
        sns.set(style="darkgrid", font_scale=1.5)

        for i_category, category in enumerate(categories):
            for i_dataset, dataset in enumerate(datasets):
                env_name = str.lower(category_names[i_category]) + dataset

                for i_mode, mode in enumerate(mode_names):
                    if (env_name, mode) in experiments:
                        data = get_all_datasets([experiments[(env_name, mode)]],
                                                None,
                                                None,
                                                None)

                        y = np.ones(10)
                        for datum in data:
                            x = np.asarray(datum['AverageTestEpRet'])
                            z = np.ones(len(x))
                            smoothed_x = np.convolve(
                                x, y, 'same') / np.convolve(z, y, 'same')
                            datum['AverageTestEpRet'] = [
                                100*get_normalized_score(env_name, score) for score in smoothed_x]

                        ax = axes[i_category, i_dataset]
                        data = pd.concat(data, ignore_index=True)
                        sns.lineplot(data=data, x='Epoch', y='AverageTestEpRet',
                                     ci='sd', estimator=getattr(np, 'mean'),
                                     ax=ax, label=mode, legend=False)
                        ax.set_ylabel('Performance')
                        # plt.legend(loc='best', ncol=1, handlelength=1,
                        #            borderaxespad=0., prop={'size': 13})


        handles, labels = ax.get_legend_handles_labels()
        labels = [label.replace('-', ' ') for label in labels]
        f.legend(handles, labels, loc='lower center', ncol=len(labels))

        pad = 5

        for ax, col in zip(axes[0], [name[1:-3].replace('-', ' ') for name in datasets]):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        for ax, row in zip(axes[:, 0], category_names):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

        plt.tight_layout(pad=0.1)
        # f.subplots_adjust(left=0.15, top=0.95)

        plt.show()