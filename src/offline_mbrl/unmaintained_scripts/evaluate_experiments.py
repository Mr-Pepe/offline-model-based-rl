import csv
import os
import os.path as osp

import d4rl
import torch

from offline_mbrl.utils.env_name_from_exp_name import get_env_name_from_experiment_name
from offline_mbrl.utils.mode_from_exp_name import get_mode_from_experiment_name
from offline_mbrl.utils.modes import PENALTY_MODES


def print_warning(text, print_args):
    print(("{}" + text + "{}").format("\033[93m", *print_args, "\033[0m"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    args = parser.parse_args()

    all_exp_dir = args.logdir

    exp_names = [
        name
        for name in os.listdir(all_exp_dir)
        if osp.isdir(osp.join(all_exp_dir, name))
    ]
    exp_names.sort()

    print()
    last_env_name = ""

    for exp_name in exp_names:
        exp_dir = osp.join(all_exp_dir, exp_name)
        env_name = get_env_name_from_experiment_name(exp_name)
        mode = get_mode_from_experiment_name(exp_name)

        if last_env_name == "":
            last_env_name = env_name

        trial_names = [
            name for name in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, name))
        ]

        if len(trial_names) != 6:
            print_warning(
                "Only {} trials for {} {}", (len(trial_names), env_name, mode)
            )

        # Check that seeds 0 to 5 were run
        for seed in range(5):
            if seed not in [int(name[-1]) for name in trial_names]:
                print_warning("Seed {} not in {} {}", (seed, env_name, mode))

        trial_performances = []

        for trial_name in trial_names:
            seed = int(trial_name[-1])
            with open(
                osp.join(exp_dir, trial_name, "progress.txt"), encoding="utf-8"
            ) as f:
                trial_log = list(csv.DictReader(f, delimiter="\t"))

                # Check that trial was run for 50 or 100 epochs
                if (
                    trial_log[-1]["Epoch"] != "50"
                    and trial_log[-1]["Epoch"] != "100"
                    and trial_log[-1]["Epoch"] != "200"
                ):
                    print_warning(
                        "{} {} seed {} ran for {} epochs only",
                        (env_name, mode, seed, trial_log[-1]["Epoch"]),
                    )

                trial_performances.append(
                    [
                        d4rl.get_normalized_score(
                            env_name, float(trial_log[i]["AverageTestEpRet"])
                        )
                        * 100
                        for i in range(len(trial_log))
                    ]
                )

        if len(trial_performances) != 6:
            print_warning(
                "Couldn't read all trial performances for {} {}", (env_name, mode)
            )

        trial_performances_tensor = torch.as_tensor(trial_performances).transpose(0, 1)

        mean_performances = trial_performances_tensor.mean(dim=1)
        std_performances = trial_performances_tensor.std(dim=1)

        if last_env_name != env_name:
            print("")

        tab = "\t"
        if mode in PENALTY_MODES:
            tab += "\t"

        max_performance = mean_performances.max(dim=0).values
        arg_max_performance = mean_performances.max(dim=0).indices
        mean_at_max_performance = std_performances[arg_max_performance]

        print(
            f"{env_name} {mode}:  "
            f"{tab}{mean_performances[-1]:.2f} +- {std_performances[-1]:.2f}\t"
            f"({max_performance:.2f} +- {mean_at_max_performance:.2f}  "
            f"@{arg_max_performance})"
        )

        last_env_name = env_name
