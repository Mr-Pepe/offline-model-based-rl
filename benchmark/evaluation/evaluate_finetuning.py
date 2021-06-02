import d4rl
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
import torch
from benchmark.evaluation.plot import get_all_datasets, make_plots, plot_data


if __name__ == "__main__":
    logdir = [
        ['/home/felipe/Projects/thesis-code/data/finetuning/halfcheetah/aleatoric-penalty/halfcheetah-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/halfcheetah/aleatoric-partitioning/halfcheetah-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/halfcheetah/epistemic-penalty/halfcheetah-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/halfcheetah/epistemic-partitioning/halfcheetah-medium-v2-sac-pretrained-50000samples'],
        ['/home/felipe/Projects/thesis-code/data/finetuning/hopper/aleatoric-penalty/hopper-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/hopper/aleatoric-partitioning/hopper-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/hopper/epistemic-penalty/hopper-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/hopper/epistemic-partitioning/hopper-medium-v2-sac-pretrained-50000samples', ],
        ['/home/felipe/Projects/thesis-code/data/finetuning/walker2d/aleatoric-penalty/walker2d-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/walker2d/aleatoric-partitioning/walker2d-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/walker2d/epistemic-penalty/walker2d-medium-v2-sac-pretrained-50000samples',
         '/home/felipe/Projects/thesis-code/data/finetuning/walker2d/epistemic-partitioning/walker2d-medium-v2-sac-pretrained-50000samples', ],
    ]

    env_names = ['halfcheetah-medium-v2',
                 'hopper-medium-v2',
                 'walker2d-medium-v2',
                 ]

    legend = [
        'aleatoric penalty',
        'aleatoric partitioning',
        'epistemic penalty',
        'epistemic partitioning'
    ]

    titles = ['Halfcheetah', 'Hopper', 'Walker2d']

    f, axes = plt.subplots(1, 3, figsize=(8, 3.5))

    for i_axis, ax in enumerate(axes):
        def normalize_score(x):
            return d4rl.get_normalized_score(env_names[i_axis], x)*100

        plt.sca(ax)
        data = get_all_datasets(logdir[i_axis], legend=legend)
        for datum in data:
            datum['AverageTestEpRet'] = [normalize_score(
                x) for x in datum['AverageTestEpRet']]
        condition = 'Condition1'
        estimator = getattr(np, 'mean')
        plot_data(data, xaxis='Epoch', value='AverageTestEpRet', condition=condition,
                  smooth=15, estimator=estimator)
        ax.grid(b=True, alpha=0.5, linestyle="--")
        ax.get_legend().remove()
        ax.set_title(titles[i_axis], fontsize=14)
        ax.set_xticks([0, 50, 100])
        ax.set_ylabel(None)

    axes[0].set_ylabel('Performance')

    if axes[0].get_legend_handles_labels()[1] != axes[1].get_legend_handles_labels()[1] or axes[0].get_legend_handles_labels()[1] != axes[2].get_legend_handles_labels()[1]:
        raise AssertionError("Legend handles not identical")

    f.legend(*axes[0].get_legend_handles_labels(),
             loc='lower center', prop={'size': 12}, ncol=2)

    f.subplots_adjust(top=0.919,
                      bottom=0.314,
                      left=0.086,
                      right=0.977,
                      hspace=0.2,
                      wspace=0.291)
    plt.show()
