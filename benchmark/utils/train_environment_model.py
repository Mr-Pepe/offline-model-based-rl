import numpy as np
from torch import nn
import torch
from torch.optim.adam import Adam


def train_environment_model(model, data, lr=1e-2, batch_size=1024):

    lr = 1e-2
    batch_size = 1024

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for i in range(2000):
        batch = data.sample_batch(batch_size)
        x = torch.cat((batch['obs'], batch['act']), dim=1)
        y = torch.cat((batch['obs2'], batch['rew'].unsqueeze(1)), dim=1)
        optim.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print("Loss: {}".format(loss))
        losses.append(loss.item())
        loss.backward()
        optim.step()

    return losses[-1]