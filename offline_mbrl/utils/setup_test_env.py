import numpy as np


def setup_test_env(env):
    if env.spec is not None:
        if env.spec.id == "antmaze-medium-diverse-v0":
            start_state = ANTMAZE_MEDIUM_DIVERSE_START_STATES[
                np.random.choice(len(ANTMAZE_MEDIUM_DIVERSE_START_STATES), 1)[0]
            ]
            o = env.reset()
            o = np.concatenate((start_state, o[2:15], o[15:]))
            env.set_state(o[:15], o[15:])
            return o
        elif env.spec.id == "antmaze-umaze-v0":
            start_state = ANTMAZE_UMAZE_START_STATES[
                np.random.choice(len(ANTMAZE_UMAZE_START_STATES), 1)[0]
            ]
            o = env.reset()
            o = np.concatenate((start_state, o[2:15], o[15:]))
            env.set_state(o[:15], o[15:])
            return o

    return env.reset()


ANTMAZE_MEDIUM_DIVERSE_START_STATES = [
    (0, 0),
    (4, 4),
    (4, 8),
    (8, 8),
    (12, 8),
    (12, 12),
    (16, 12),
    (20, 12),
    (20, 16),
]

ANTMAZE_UMAZE_START_STATES = [(0, 0), (4, 0), (8, 0), (8, 4), (8, 8), (4, 8)]
