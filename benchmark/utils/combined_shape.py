import numpy as np


def combined_shape(length, shape=None):
    # Based on https://spinningup.openai.com

    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
