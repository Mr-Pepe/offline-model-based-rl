import torch


def deterministic_loss(x, y, model, i_network=0, terminal_loss_factor=1):
    y_pred, _, _, _, _ = model(x, i_network)
    loss = torch.square(y_pred - y)
    loss[:, -1] = loss[:, -1]*terminal_loss_factor
    return loss.mean()


def probabilistic_loss(x, y, model, i_network=0, only_mse=False,
                       terminal_loss_factor=1):
    _, mean, logvar, max_logvar, min_logvar = model(x, i_network)
    inv_var = torch.exp(-logvar)

    if only_mse:
        loss = torch.square(mean - y)
        loss[:, -1] = loss[:, -1]*terminal_loss_factor
        return loss.mean()
    else:
        se = torch.square(mean - y)
        se[:, -1] = se[:, -1]*terminal_loss_factor
        mse_loss = (se * inv_var).mean()

        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.sum() - 0.01 * min_logvar.sum()

        # print("{:.3f}, {:.3f}, {:.3f}".format(
        #     mse_loss, var_loss, var_bound_loss))
        return mse_loss + var_loss + var_bound_loss
