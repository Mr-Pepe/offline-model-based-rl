from torch._C import dtype
from benchmark.models.environment_model import EnvironmentModel
import torch.nn as nn
import torch
from torch.optim.adam import Adam


def test_is_nn_module():
    assert issubclass(EnvironmentModel, nn.Module)


def test_takes_state_as_input_and_outputs_same_dimensions():
    model = EnvironmentModel(5, 5)

    tensor_size = (3, 5)
    input = torch.rand(tensor_size)
    output = model(input)

    assert output.shape == input.shape


def test_can_overfit_on_single_sample():
    model = EnvironmentModel(1, 1)

    x = torch.as_tensor([3], dtype=torch.float32)
    y = torch.as_tensor([5], dtype=torch.float32)
    lr = 1e-3

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert float(y - y_pred) < 1e-5


def test_can_overfit_on_batch():

    model = EnvironmentModel(10, 10)

    tensor_size = (10, 10)

    x = torch.rand(tensor_size)
    y = torch.rand(tensor_size)
    lr = 1e-3

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_pred = 0

    for i in range(1000):
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        loss.backward()
        optim.step()

    assert float((y - y_pred).mean().sum()) < 1e-5
