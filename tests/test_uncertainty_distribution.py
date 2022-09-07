import pytest

from offline_mbrl.utils.uncertainty_distribution import get_uncertainty_distribution


def test_raises_error_for_unknown_mode() -> None:
    with pytest.raises(ValueError, match=r"Mode must be in .* but got 'mode'."):
        get_uncertainty_distribution("", "mode")
