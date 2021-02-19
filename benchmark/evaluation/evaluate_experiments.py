from benchmark.utils.modes import MODES, PENALTY_MODES
from benchmark.utils.envs import HALF_CHEETAH_ENVS, HOPPER_ENVS, WALKER_ENVS
import os
import os.path as osp
import csv
import torch
import d4rl
import warnings


def get_env_name(exp_name):
    for env_name in HOPPER_ENVS + HALF_CHEETAH_ENVS + WALKER_ENVS:
        if env_name in exp_name:
            return env_name

    return None


def get_mode(exp_name):
    for mode in MODES:
        if mode in exp_name:
            return mode

    return None


def print_warning(text, args):
    print(("{}" + text + "{}").format('\033[93m', *args, '\033[0m'))


all_exp_dir = "/home/felipe/Projects/thesis-code/data/experiments/"

exp_names = [name for name in os.listdir(
    all_exp_dir) if osp.isdir(osp.join(all_exp_dir, name))]
exp_names.sort()

print()
last_env_name = ''

for exp_name in exp_names:
    exp_dir = osp.join(all_exp_dir, exp_name)
    env_name = get_env_name(exp_name)
    mode = get_mode(exp_name)

    if last_env_name == '':
        last_env_name = env_name

    trial_names = [name for name in os.listdir(
        exp_dir) if osp.isdir(osp.join(exp_dir, name))]

    if len(trial_names) != 6:
        print_warning("Only {} trials for {} {}",
                      (len(trial_names), env_name, mode))

    # Check that seeds 0 to 5 were run
    for seed in range(5):
        if seed not in [int(name[-1]) for name in trial_names]:
            print_warning("Seed {} not in {} {}",
                          (seed, env_name, mode))

    trial_performances = []

    for trial_name in trial_names:
        seed = int(trial_name[-1])
        with open(osp.join(exp_dir, trial_name, 'progress.txt'), 'r') as f:
            test_returns = list(csv.DictReader(f, delimiter='\t'))

            # Check that trial was run for 50 epochs
            if test_returns[-1]['Epoch'] != '50':
                print_warning("{} {} seed {} ran for {} epochs only",
                              (env_name, mode, seed, test_returns[-1]['Epoch']))

            trial_performances.append(d4rl.get_normalized_score(
                env_name, float(test_returns[-1]['AverageTestEpRet']))*100)

    if len(trial_performances) != 6:
        print_warning("Couldn't read all trial performances for {} {}",
                      (env_name, mode))

    trial_performances = torch.as_tensor(trial_performances)

    mean_performance = trial_performances.mean()
    std_performance = trial_performances.std()

    if last_env_name != env_name:
        print('')

    tab = '\t'
    if mode in PENALTY_MODES:
        tab += '\t'

    print("{} {}:  {}{:.2f} +- {:.2f}".format(env_name,
                                             mode,
                                             tab,
                                             mean_performance,
                                             std_performance))

    last_env_name = env_name
