import torch


def deterministic_loss(x, y, model):
    y_pred, _, _ = model(x)
    return torch.square(y_pred - y).mean()


def probabilistic_loss(x, y, model, only_mse=False):
    _, mean, logvar = model(x)
    inv_var = torch.exp(-logvar)

    if only_mse:
        return torch.square(mean - y).mean()
    else:
        mse_loss = (torch.square(mean - y) * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * model.max_logvar.sum() - \
            0.01*model.min_logvar.sum()

        # print("{:.3f}, {:.3f}, {:.3f}".format(
        #     mse_loss, var_loss, var_bound_loss))
        return mse_loss + var_loss + var_bound_loss
