from benchmark.utils.loss_functions import deterministic_loss, \
    probabilistic_loss
import torch
from benchmark.models.environment_model import EnvironmentModel


def test_terminal_loss_factor_increases_deterministic_loss():
    torch.manual_seed(0)

    model = EnvironmentModel(3, 3, type='deterministic')

    x = torch.rand((3, 6), dtype=torch.float32)
    y = torch.rand((3, 5), dtype=torch.float32)

    a = deterministic_loss(x, y, model)
    b = deterministic_loss(x, y, model, terminal_loss_factor=5)

    assert a < b


def test_terminal_loss_factor_increases_probabilistic_loss():
    torch.manual_seed(0)

    model = EnvironmentModel(3, 3, type='probabilistic')

    x = torch.rand((3, 6), dtype=torch.float32)
    y = torch.rand((3, 5), dtype=torch.float32)

    a = probabilistic_loss(x, y, model)
    b = probabilistic_loss(x, y, model, terminal_loss_factor=5)

    assert a < b
