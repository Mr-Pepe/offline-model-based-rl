import os
import os.path as osp


all_exp_dir = "/home/felipe/Projects/thesis-code/data/experiments/"

exp_names = [name for name in os.listdir(all_exp_dir) if osp.isdir(osp.join(all_exp_dir, name))]

for exp_name in exp_names:
    exp_dir = osp.join(all_exp_dir, exp_name)

    trial_names = [name for name in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, name))]

    assert len(trial_names) == 6

    # Check that seeds 0 to 5 were run
    for seed in range(5):
        assert seed in [int(name[-1]) for name in trial_names]

    # Check that all trials were run for 50 epochs
    for trial_dir in os.listdir(exp_dir):
        with open(osp.join(exp_dir, trial_dir, 'progress.txt'), 'r') as f:
            assert '50' == f.readlines()[-1][:2]
