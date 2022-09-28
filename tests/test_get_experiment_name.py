from offline_mbrl.schemas import TrainerConfiguration
from offline_mbrl.utils.get_experiment_name import get_experiment_name
from offline_mbrl.utils.modes import SAC


def test_get_experiment_name() -> None:
    config = TrainerConfiguration()
    mode = SAC

    assert get_experiment_name(config, mode) == "hopper-medium-replay-v2-sac"
