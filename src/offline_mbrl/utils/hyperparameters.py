#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains optimizied hyperparameters for the different environments."""

from offline_mbrl.utils.envs import *  # pylint: disable=wildcard-import, unused-wildcard-import
from offline_mbrl.utils.modes import (
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY,
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY,
)

HYPERPARAMS = {
    ALEATORIC_PARTITIONING: {
        HALF_CHEETAH_RANDOM_V2: (19, 0.36515),
        HALF_CHEETAH_MEDIUM_V2: (18, 2.23),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (15, 0.69141),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (7, 1.0944),
        HALF_CHEETAH_EXPERT_V2: (5, 0.31109),
        HOPPER_RANDOM_V2: (9, 0.00436),
        HOPPER_MEDIUM_V2: (19, 0.0131),
        HOPPER_MEDIUM_REPLAY_V2: (9, 0.0483),
        HOPPER_MEDIUM_EXPERT_V2: (9, 0.0473),
        HOPPER_EXPERT_V2: (20, 0.00145),
        WALKER_RANDOM_V2: (17, 0.40423),
        WALKER_MEDIUM_V2: (10, 1.8323),
        WALKER_MEDIUM_REPLAY_V2: (20, 1.1510),
        WALKER_MEDIUM_EXPERT_V2: (15, 2.4947),
        WALKER_EXPERT_V2: (8, 1.3946),
    },
    EPISTEMIC_PARTITIONING: {
        HALF_CHEETAH_RANDOM_V2: (18, 4.2511),
        HALF_CHEETAH_MEDIUM_V2: (13, 3.733),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (20, 3.5232),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (17, 1.4370),
        HALF_CHEETAH_EXPERT_V2: (7, 1.5848),
        HOPPER_RANDOM_V2: (11, 0.20551),
        HOPPER_MEDIUM_V2: (18, 0.72861),
        HOPPER_MEDIUM_REPLAY_V2: (20, 0.599),
        HOPPER_MEDIUM_EXPERT_V2: (20, 1.0881),
        HOPPER_EXPERT_V2: (18, 0.19435),
        WALKER_RANDOM_V2: (15, 1.1804),
        WALKER_MEDIUM_V2: (11, 1.5975),
        WALKER_MEDIUM_REPLAY_V2: (4, 3.12812),
        WALKER_MEDIUM_EXPERT_V2: (10, 1.9587),
        WALKER_EXPERT_V2: (19, 3.0755),
    },
    ALEATORIC_PENALTY: {
        HALF_CHEETAH_RANDOM_V2: (17, 0),
        HALF_CHEETAH_MEDIUM_V2: (3, 1.8882),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (17, 0.69304),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (2, 8.1968),
        HALF_CHEETAH_EXPERT_V2: (11, 10.944),
        HOPPER_RANDOM_V2: (10, 15.398),
        HOPPER_MEDIUM_V2: (15, 300),
        HOPPER_MEDIUM_REPLAY_V2: (8, 48.06),
        HOPPER_MEDIUM_EXPERT_V2: (15, 380),
        HOPPER_EXPERT_V2: (40, 438.51),
        WALKER_RANDOM_V2: (20, 0.1491),
        WALKER_MEDIUM_V2: (9, 1.51),
        WALKER_MEDIUM_REPLAY_V2: (20, 1.32213),
        WALKER_MEDIUM_EXPERT_V2: (19, 2.3841),
        WALKER_EXPERT_V2: (17, 1.8705),
    },
    EPISTEMIC_PENALTY: {
        HALF_CHEETAH_RANDOM_V2: (17, 0.22774),
        HALF_CHEETAH_MEDIUM_V2: (4, 1e-15),
        HALF_CHEETAH_MEDIUM_REPLAY_V2: (15, 5),
        HALF_CHEETAH_MEDIUM_EXPERT_V2: (1, 5),
        HALF_CHEETAH_EXPERT_V2: (13, 0.68733),
        HOPPER_RANDOM_V2: (5, 3.9639),
        HOPPER_MEDIUM_V2: (15, 35),
        HOPPER_MEDIUM_REPLAY_V2: (15, 35),
        HOPPER_MEDIUM_EXPERT_V2: (15, 45),
        HOPPER_EXPERT_V2: (15, 30),
        WALKER_RANDOM_V2: (15, 2.5),
        WALKER_MEDIUM_V2: (15, 4.3),
        WALKER_MEDIUM_REPLAY_V2: (15, 4.3),
        WALKER_MEDIUM_EXPERT_V2: (15, 4.3),
        WALKER_EXPERT_V2: (15, 2.5),
    },
}
"""Tuned hyperparameters.

Contains tuples of (virtual rollout length, pessimism/ OOD threshold).
"""
