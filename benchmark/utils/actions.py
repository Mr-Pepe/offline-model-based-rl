from enum import IntEnum


class Actions(IntEnum):
    TRAIN_MODEL = 0
    UPDATE_AGENT = 1
    RANDOM_ACTION = 2
    GENERATE_ROLLOUTS = 3
