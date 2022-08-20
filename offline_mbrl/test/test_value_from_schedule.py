import pytest
from offline_mbrl.utils.value_from_schedule import get_value_from_schedule


@pytest.mark.fast
def test_rollout_length_from_schedule():
    schedule = [1, 15, 20, 100]

    assert get_value_from_schedule(schedule, 1) == 1
    assert get_value_from_schedule(schedule, 20) == 1
    assert get_value_from_schedule(schedule, 100) == 15
    assert get_value_from_schedule(schedule, 120) == 15
    assert get_value_from_schedule(schedule, 50) == 6
