from benchmark.utils.modes import MODES


def get_mode(exp_name):
    for mode in MODES:
        if mode in exp_name:
            return mode

    return None
