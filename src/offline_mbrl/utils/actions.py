from enum import IntEnum


class Actions(IntEnum):
    """Actions that can be performed during an iteration of the training loop.

    Used for logging what happened during training and testing the training loop.
    """

    TRAIN_MODEL = 0
    UPDATE_AGENT = 1
    RANDOM_ACTION = 2
    GENERATE_ROLLOUTS = 3
    INTERACT_WITH_ENV = 4
