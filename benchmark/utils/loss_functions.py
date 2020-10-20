import torch


def deterministic_loss(x, y, model):
    y_pred = model(x)
    return torch.square(y_pred - y)


def probabilistic_loss(x, y, model, only_mse=False):
    mean, logvar, max_logvar, min_logvar = model.predict_mean_and_logvar(x)
    inv_var = torch.exp(-logvar)

    if only_mse:
        std = torch.exp(0.5*logvar)
        y_pred = torch.normal(mean, std)
        return torch.square(y - y_pred).mean()
    else:
        mse_loss = (torch.square(
            mean - y) * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01*max_logvar.sum() - 0.01*min_logvar.sum()
        return mse_loss + var_loss + var_bound_loss
