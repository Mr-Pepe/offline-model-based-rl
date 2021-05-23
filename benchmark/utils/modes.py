EXPLICIT_PARTITIONING = 'explicit-partitioning'
ALEATORIC_PARTITIONING = 'aleatoric-partitioning'
EPISTEMIC_PARTITIONING = 'epistemic-partitioning'
EXPLICIT_PENALTY = 'explicit-penalty'
ALEATORIC_PENALTY = 'aleatoric-penalty'
EPISTEMIC_PENALTY = 'epistemic-penalty'
OFFLINE_EXPLORATION_PENALTY = 'offline-exploration-penalty'
OFFLINE_EXPLORATION_PARTITIONING = 'offline-exploration-partitioning'
UNDERESTIMATION = 'underestimation'
BEHAVIORAL_CLONING = 'behavioral-cloning'
CQL = 'cql'
COPYCAT = 'copycat'
SAC = 'sac'
MBPO = 'mbpo'
SURVIVAL = 'survival'

MODES = [
    EXPLICIT_PARTITIONING,
    ALEATORIC_PARTITIONING,
    EPISTEMIC_PARTITIONING,
    EXPLICIT_PENALTY,
    ALEATORIC_PENALTY,
    EPISTEMIC_PENALTY,
    UNDERESTIMATION,
    BEHAVIORAL_CLONING,
    CQL,
    COPYCAT,
    OFFLINE_EXPLORATION_PENALTY,
    OFFLINE_EXPLORATION_PARTITIONING,
    SAC,
    MBPO,
    SURVIVAL
]

PENALTY_MODES = [
    EXPLICIT_PENALTY,
    ALEATORIC_PENALTY,
    EPISTEMIC_PENALTY
]

PARTITIONING_MODES = [
    EXPLICIT_PARTITIONING,
    ALEATORIC_PARTITIONING,
    EPISTEMIC_PARTITIONING
]

ALEATORIC_MODES = [
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY
]

EPISTEMIC_MODES = [
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY
]

EXPLICIT_MODES = [
    EXPLICIT_PARTITIONING,
    EXPLICIT_PENALTY
]
