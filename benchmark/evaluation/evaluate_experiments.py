from benchmark.utils.modes import PENALTY_MODES
from benchmark.utils.env_name_from_exp_name import get_env_name
from benchmark.utils.mode_from_exp_name import get_mode
from benchmark.utils.print_warning import print_warning
import os
import os.path as osp
import csv
import torch
import d4rl


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str, default='')
    args = parser.parse_args()

    all_exp_dir = args.logdir

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
                trial_log = list(csv.DictReader(f, delimiter='\t'))

                # Check that trial was run for 50 or 100 epochs
                if trial_log[-1]['Epoch'] != '50' and \
                    trial_log[-1]['Epoch'] != '100' and \
                        trial_log[-1]['Epoch'] != '200':
                    print_warning("{} {} seed {} ran for {} epochs only",
                                  (env_name, mode, seed, trial_log[-1]['Epoch']))

                trial_performances.append([d4rl.get_normalized_score(
                    env_name, float(trial_log[i]['AverageTestEpRet']))*100 for i in range(len(trial_log))])

        if len(trial_performances) != 6:
            print_warning("Couldn't read all trial performances for {} {}",
                          (env_name, mode))

        trial_performances = torch.as_tensor(
            trial_performances).transpose(0, 1)

        mean_performances = trial_performances.mean(dim=1)
        std_performances = trial_performances.std(dim=1)

        if last_env_name != env_name:
            print('')

        tab = '\t'
        if mode in PENALTY_MODES:
            tab += '\t'

        max_performance = mean_performances.max(dim=0).values
        arg_max_performance = mean_performances.max(dim=0).indices
        mean_at_max_performance = std_performances[arg_max_performance]

        print("{} {}:  {}{:.2f} +- {:.2f}\t({:.2f} +- {:.2f}  @{})".format(
            env_name,
            mode,
            tab,
            mean_performances[-1],
            std_performances[-1],
            max_performance,
            mean_at_max_performance,
            arg_max_performance))

        last_env_name = env_name
