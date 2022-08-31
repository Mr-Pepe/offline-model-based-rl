#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module defines the environments available in this package."""

from offline_mbrl.utils.modes import (
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY,
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY,
)

HOPPER_RANDOM_V2 = "hopper-random-v2"
HOPPER_MEDIUM_V2 = "hopper-medium-v2"
HOPPER_EXPERT_V2 = "hopper-expert-v2"
HOPPER_MEDIUM_REPLAY_V2 = "hopper-medium-replay-v2"
HOPPER_MEDIUM_EXPERT_V2 = "hopper-medium-expert-v2"

HOPPER_ENVS = [
    HOPPER_RANDOM_V2,
    HOPPER_MEDIUM_V2,
    HOPPER_EXPERT_V2,
    HOPPER_MEDIUM_REPLAY_V2,
    HOPPER_MEDIUM_EXPERT_V2,
]


HALF_CHEETAH_RANDOM_V2 = "halfcheetah-random-v2"
HALF_CHEETAH_MEDIUM_V2 = "halfcheetah-medium-v2"
HALF_CHEETAH_EXPERT_V2 = "halfcheetah-expert-v2"
HALF_CHEETAH_MEDIUM_REPLAY_V2 = "halfcheetah-medium-replay-v2"
HALF_CHEETAH_MEDIUM_EXPERT_V2 = "halfcheetah-medium-expert-v2"

HALF_CHEETAH_ENVS = [
    HALF_CHEETAH_RANDOM_V2,
    HALF_CHEETAH_MEDIUM_V2,
    HALF_CHEETAH_EXPERT_V2,
    HALF_CHEETAH_MEDIUM_REPLAY_V2,
    HALF_CHEETAH_MEDIUM_EXPERT_V2,
]


WALKER_RANDOM_V2 = "walker2d-random-v2"
WALKER_MEDIUM_V2 = "walker2d-medium-v2"
WALKER_EXPERT_V2 = "walker2d-expert-v2"
WALKER_MEDIUM_REPLAY_V2 = "walker2d-medium-replay-v2"
WALKER_MEDIUM_EXPERT_V2 = "walker2d-medium-expert-v2"


WALKER_ENVS = [
    WALKER_RANDOM_V2,
    WALKER_MEDIUM_V2,
    WALKER_EXPERT_V2,
    WALKER_MEDIUM_REPLAY_V2,
    WALKER_MEDIUM_EXPERT_V2,
]

ENV_CATEGORIES = {
    "hopper": HOPPER_ENVS,
    "half_cheetah": HALF_CHEETAH_ENVS,
    "walker2d": WALKER_ENVS,
}

ALL_ENVS = [*HOPPER_ENVS, *HALF_CHEETAH_ENVS, *WALKER_ENVS]


# Hyperparameters
# (rollouts, rollout length, pessimism/ OOD threshold)
HYPERPARAMS = {
    ALEATORIC_PARTITIONING: {
        HALF_CHEETAH_RANDOM_V2: (50, 19, 0.36515),
        HALF_CHEETAH_MEDIUM_V2: (50, 18, 2.23),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (50, 15, 0.69141),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (50, 7, 1.0944),
        HALF_CHEETAH_EXPERT_V2: (50, 5, 0.31109),
        HOPPER_RANDOM_V2: (50, 9, 0.00436),
        HOPPER_MEDIUM_V2: (50, 19, 0.0131),
        HOPPER_MEDIUM_REPLAY_V2: (50, 9, 0.0483),
        HOPPER_MEDIUM_EXPERT_V2: (50, 9, 0.0473),
        HOPPER_EXPERT_V2: (50, 20, 0.00145),
        WALKER_RANDOM_V2: (50, 17, 0.40423),
        WALKER_MEDIUM_V2: (50, 10, 1.8323),
        WALKER_MEDIUM_REPLAY_V2: (50, 20, 1.1510),
        WALKER_MEDIUM_EXPERT_V2: (50, 15, 2.4947),
        WALKER_EXPERT_V2: (50, 8, 1.3946),
    },
    EPISTEMIC_PARTITIONING: {
        HALF_CHEETAH_RANDOM_V2: (50, 18, 4.2511),
        HALF_CHEETAH_MEDIUM_V2: (50, 13, 3.733),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (50, 20, 3.5232),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (50, 17, 1.4370),
        HALF_CHEETAH_EXPERT_V2: (50, 7, 1.5848),
        HOPPER_RANDOM_V2: (50, 11, 0.20551),
        HOPPER_MEDIUM_V2: (50, 18, 0.72861),
        HOPPER_MEDIUM_REPLAY_V2: (50, 20, 0.599),
        HOPPER_MEDIUM_EXPERT_V2: (50, 20, 1.0881),
        HOPPER_EXPERT_V2: (50, 18, 0.19435),
        WALKER_RANDOM_V2: (50, 15, 1.1804),
        WALKER_MEDIUM_V2: (50, 11, 1.5975),
        WALKER_MEDIUM_REPLAY_V2: (50, 4, 3.12812),
        WALKER_MEDIUM_EXPERT_V2: (50, 10, 1.9587),
        WALKER_EXPERT_V2: (50, 19, 3.0755),
    },
    ALEATORIC_PENALTY: {
        HALF_CHEETAH_RANDOM_V2: (50, 17, 0),
        HALF_CHEETAH_MEDIUM_V2: (50, 3, 1.8882),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (50, 17, 0.69304),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (50, 2, 8.1968),
        HALF_CHEETAH_EXPERT_V2: (50, 11, 10.944),
        HOPPER_RANDOM_V2: (50, 10, 15.398),
        HOPPER_MEDIUM_V2: (50, 15, 300),
        HOPPER_MEDIUM_REPLAY_V2: (50, 8, 48.06),
        HOPPER_MEDIUM_EXPERT_V2: (50, 15, 380),
        HOPPER_EXPERT_V2: (50, 40, 438.51),
        WALKER_RANDOM_V2: (50, 20, 0.1491),
        WALKER_MEDIUM_V2: (50, 9, 1.51),
        WALKER_MEDIUM_REPLAY_V2: (50, 20, 1.32213),
        WALKER_MEDIUM_EXPERT_V2: (50, 19, 2.3841),
        WALKER_EXPERT_V2: (50, 17, 1.8705),
    },
    EPISTEMIC_PENALTY: {
        HALF_CHEETAH_RANDOM_V2: (50, 17, 0.22774),
        HALF_CHEETAH_MEDIUM_V2: (50, 4, 1e-15),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (50, 15, 5),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (50, 1, 5),
        HALF_CHEETAH_EXPERT_V2: (50, 13, 0.68733),
        HOPPER_RANDOM_V2: (50, 5, 3.9639),
        HOPPER_MEDIUM_V2: (50, 15, 35),
        HOPPER_MEDIUM_REPLAY_V2: (50, 15, 35),
        HOPPER_MEDIUM_EXPERT_V2: (50, 15, 45),
        HOPPER_EXPERT_V2: (50, 15, 30),
        WALKER_RANDOM_V2: (50, 15, 2.5),
        WALKER_MEDIUM_V2: (50, 15, 4.3),
        WALKER_MEDIUM_REPLAY_V2: (50, 15, 4.3),
        WALKER_MEDIUM_EXPERT_V2: (50, 15, 4.3),
        WALKER_EXPERT_V2: (50, 15, 2.5),
    },
}
