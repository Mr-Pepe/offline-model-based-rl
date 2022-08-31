import pytest

from offline_mbrl.utils.env_name_from_exp_name import get_env_name_from_experiment_name


def test_env_name_gets_retrieved_from_experiment_name() -> None:
    assert (
        get_env_name_from_experiment_name("blablahopper-random-v2-blabla")
        == "hopper-random-v2"
    )


def test_error_is_raised_if_env_name_not_found() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Failed to retrieve environment name from experiment name 'abc'. "
            "The experiment name did not contain any of the following environment "
            "names: "
        ),
    ):
        get_env_name_from_experiment_name("abc")
