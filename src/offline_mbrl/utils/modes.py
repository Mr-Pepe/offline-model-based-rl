#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains the different training modes available in this package."""


ALEATORIC_PARTITIONING = "aleatoric-partitioning"
EPISTEMIC_PARTITIONING = "epistemic-partitioning"
ALEATORIC_PENALTY = "aleatoric-penalty"
EPISTEMIC_PENALTY = "epistemic-penalty"
BEHAVIORAL_CLONING = "behavioral-cloning"
SAC = "sac"
MBPO = "mbpo"

ALL_MODES = [
    ALEATORIC_PARTITIONING,
    EPISTEMIC_PARTITIONING,
    ALEATORIC_PENALTY,
    EPISTEMIC_PENALTY,
    BEHAVIORAL_CLONING,
    SAC,
    MBPO,
]

PENALTY_MODES = [ALEATORIC_PENALTY, EPISTEMIC_PENALTY]

PARTITIONING_MODES = [
    ALEATORIC_PARTITIONING,
    EPISTEMIC_PARTITIONING,
]

ALEATORIC_MODES = [ALEATORIC_PARTITIONING, ALEATORIC_PENALTY]

EPISTEMIC_MODES = [EPISTEMIC_PARTITIONING, EPISTEMIC_PENALTY]
