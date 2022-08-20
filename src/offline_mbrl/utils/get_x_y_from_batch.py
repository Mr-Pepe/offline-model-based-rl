import torch


def get_x_y_from_batch(batch, device):

    x = torch.cat((batch["obs"], batch["act"]), dim=1)
    y = torch.cat((batch["obs2"], batch["rew"].unsqueeze(1)), dim=1)

    x = x.to(device)
    y = y.to(device)

    return x, y
