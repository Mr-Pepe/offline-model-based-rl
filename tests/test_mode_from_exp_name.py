import pytest

from offline_mbrl.utils.mode_from_exp_name import get_mode_from_experiment_name


@pytest.mark.fast
def test_mode_gets_retrieved_from_experiment_name() -> None:
    assert (
        get_mode_from_experiment_name("blablaaleatoric-partitioning-blabla")
        == "aleatoric-partitioning"
    )


@pytest.mark.fast
def test_error_is_raised_if_mode_not_found() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Failed to retrieve mode from experiment name 'abc'. "
            "The experiment name did not contain any of the following modes: "
        ),
    ):
        get_mode_from_experiment_name("abc")
