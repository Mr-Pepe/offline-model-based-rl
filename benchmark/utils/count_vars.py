import numpy as np


def count_vars(module):
    # Based on https://spinningup.openai.com

    return sum([np.prod(p.shape) for p in module.parameters()])
