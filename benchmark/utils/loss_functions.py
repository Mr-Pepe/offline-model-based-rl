import torch


def deterministic_loss(x, y, model, i_network=0):
    y_pred, _, _, _, _ = model(x, i_network)
    return torch.square(y_pred[:, :-1] - y).mean()


def probabilistic_loss(x, y, model, i_network=0, only_mse=False):
    _, mean, logvar, max_logvar, min_logvar = model(x, i_network)
    inv_var = torch.exp(-logvar)

    if only_mse:
        return torch.square(mean - y).mean()
    else:
        mse_loss = (torch.square(mean - y) * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.sum() - 0.01 * min_logvar.sum()

        # print("{:.3f}, {:.3f}, {:.3f}".format(
        #     mse_loss, var_loss, var_bound_loss))
        return mse_loss + var_loss + var_bound_loss
