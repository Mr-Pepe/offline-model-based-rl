#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module defines the environments available in this package."""


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
"""Hopper environments."""


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
"""Halfcheetah environments."""


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
"""Walker2D environments."""

ENV_CATEGORIES = {
    "hopper": HOPPER_ENVS,
    "half_cheetah": HALF_CHEETAH_ENVS,
    "walker2d": WALKER_ENVS,
}

ALL_ENVS = [*HOPPER_ENVS, *HALF_CHEETAH_ENVS, *WALKER_ENVS]
