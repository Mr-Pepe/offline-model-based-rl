from benchmark.utils.modes import ALEATORIC_PARTITIONING, ALEATORIC_PENALTY, BEHAVIORAL_CLONING, EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY, MBPO, SAC
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
import torch
from benchmark.evaluation.plot import make_plots


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', default=[
                        '/home/felipe/Projects/thesis-code/data/online_experiments/aleatoric_penalty',
                        '/home/felipe/Projects/thesis-code/data/online_experiments/aleatoric-partitioning',
                        '/home/felipe/Projects/thesis-code/data/online_experiments/epistemic-penalty/walker2d-medium-v2-sac-pretrained-50000samples',
                        '/home/felipe/Projects/thesis-code/data/online_experiments/epistemic-partitioning/walker2d-medium-v2-sac-pretrained-50000samples'
                        ], nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Epoch')
    parser.add_argument(
        '--value', '-y', default=['AverageTestEpRet'], nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=15)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()

    legend = [
        'aleatoric penalty',
        'aleatoric partitioning',
        'epistemic penalty',
        'epistemic partitioning'
    ]

    make_plots(args.logdir, legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)
