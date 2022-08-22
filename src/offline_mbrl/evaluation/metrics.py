import numpy as np
from scipy.stats import iqr, spearmanr


def final_performance(performances):
    """Return the final performance for a collection of training results.

    Arguments:
    performances -- n x m array with n epochs and m trials

    Return value:
    A tuple containing the mean and standard performance at the last epoch.
    """

    return (
        performances[-5:, :].mean(),
        performances[-5:, :].std(),
        iqr(performances[-5:, :].flatten()),
    )


def stability(performances):
    """Compute the training stability of a method.

    Arguments:
    performances -- n x m array with n epochs and m trials

    Return value:
    The mean spearman rank correlation across trials.
    """

    coeff = np.mean(
        [
            spearmanr(range(len(performances)), performances[:, i])[0]
            for i in range(performances.shape[1])
        ]
    )

    return coeff


def efficiency(performances):
    """Compute the number of steps to reach 80% of the maximum policy performance.

    Arguments:
    performances -- n x m array with n epochs and m trials

    Return value:
    Mean number of steps to reach 80% of the maximum performance.
    """

    max_perf = performances.max(axis=0)
    min_perf = performances.min(axis=0)
    tt80 = np.zeros_like(max_perf)

    for i_trial, trial in enumerate(performances.transpose()):
        idx = trial > (
            max_perf[i_trial] - (max_perf[i_trial] - min_perf[i_trial]) * 0.2
        )
        tt80[i_trial] = [i for i, x in enumerate(idx) if x][0]

    return tt80.mean()


def estimation_quality(model_errors, uncertainty_estimates):
    """Takes one-dimensional numpy arrays for true model errors
    and uncertainty estimates and returns the exponential deviation between the two."""
    model_errors /= model_errors.max()
    uncertainty_estimates /= uncertainty_estimates.max()

    return (np.exp((model_errors - uncertainty_estimates).abs()) - 1).mean()


def pessimism(model_errors, uncertainty_estimates):
    """Takes one-dimensional numpy arrays for true model errors
    and uncertainty estimates and returns how pessimistic the model is in the range
    [-1, 1]."""
    model_errors /= model_errors.max()
    uncertainty_estimates /= uncertainty_estimates.max()

    errors = np.exp((model_errors - uncertainty_estimates).abs()) - 1

    return errors[model_errors < uncertainty_estimates].sum() / errors.sum() * 2 - 1
