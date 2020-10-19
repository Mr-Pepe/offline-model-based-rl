import numpy as np
from torch import nn
import torch
from torch.optim.adam import Adam


def train_environment_model(model, data, lr=1e-2, batch_size=1024, val_split=0.2, patience=20):
    """ Train an environment model on a replay buffer until convergence """

    device = next(model.parameters()).device

    data_size = data.size
    n_train_batches = int((data_size * (1-val_split)) // batch_size)
    n_val_batches = int((data_size * val_split) // batch_size)

    if n_train_batches == 0 or n_val_batches == 0:
        raise ValueError("Dataset not big enough to generate a train/val split with the given batch size.")

    optim = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss = 1e10
    n_bad_val_losses = 0
    avg_val_loss = 0
    avg_train_loss = 0

    while n_bad_val_losses < patience:

        avg_train_loss = 0

        for i in range(n_train_batches):
            batch = data.sample_train_batch(batch_size, val_split)
            x = torch.cat((batch['obs'], batch['act']), dim=1)
            y = torch.cat((batch['obs2'], batch['rew'].unsqueeze(1)), dim=1)

            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            avg_train_loss += loss.item()
            loss.backward()
            optim.step()

        print("Train loss: {}".format(avg_train_loss/n_train_batches))

        avg_val_loss = 0
        for i in range(n_val_batches):
            batch = data.sample_val_batch(batch_size, val_split)
            x = torch.cat((batch['obs'], batch['act']), dim=1)
            y = torch.cat((batch['obs2'], batch['rew'].unsqueeze(1)), dim=1)

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            avg_val_loss += criterion(y_pred, y).item()

        avg_val_loss /= n_val_batches

        if avg_val_loss < min_val_loss:
            n_bad_val_losses = 0
            min_val_loss = avg_val_loss
        else:
            n_bad_val_losses += 1

        print("Patience: {} Val loss: {}".format(n_bad_val_losses, avg_val_loss))

    return avg_val_loss