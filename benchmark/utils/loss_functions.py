import torch


def deterministic_loss(x, y, model, i_network=-1):
    y_pred, _, _, _, _ = model(x)

    if i_network == -1:
        return torch.square(y_pred[:, :, :-1] - y.unsqueeze(0)).mean()
    else:
        return torch.square(y_pred[i_network, :, :-1] - y.unsqueeze(0)).mean()


def probabilistic_loss(x, y, model, i_network=-1, only_mse=False, debug=False):
    _, mean, logvar, max_logvar, min_logvar = model(x)

    if i_network > -1:
        mean = mean[i_network]
        logvar = logvar[i_network]
        max_logvar = max_logvar[i_network]
        min_logvar = min_logvar[i_network]

    inv_var = torch.exp(-logvar)

    if only_mse:
        return torch.square(mean - y).mean()
    else:
        mse_loss = torch.square(mean - y)
        mse_inv_var_loss = (mse_loss * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = torch.clamp(
            0.01 * max_logvar.sum() - 0.01 * min_logvar.sum(),
            0,
            1e10)

        if debug:
            print("MSE: {:.5f}, MSE + INV VAR: {:.5f} VAR: {:.5f}, BOUNDS: {:.5f}, MAX LOGVAR: {:.5f}, MIN LOGVAR: {:.5f}".format(
                mse_loss.mean().item(),
                mse_inv_var_loss.item(),
                var_loss.item(),
                var_bound_loss.item(),
                max_logvar.mean().item(),
                min_logvar.mean().item()
            ), end='\r')

        return mse_inv_var_loss + var_loss + var_bound_loss
