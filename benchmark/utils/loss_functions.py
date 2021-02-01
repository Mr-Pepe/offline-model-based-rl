import torch


def deterministic_loss(x, y, model, i_network=-1):
    y_pred, _, _, _, _ = model(x)

    if i_network == -1:
        return torch.square(y_pred - y.unsqueeze(0)).mean()
    else:
        return torch.square(y_pred[i_network] - y.unsqueeze(0)).mean()


def probabilistic_loss(x, y, model, i_network=-1, only_mse=False, debug=False, no_reward=False,
                       only_var_loss=False):
    _, mean, logvar, max_logvar, min_logvar = model(x)

    if no_reward:
        x = x[:, :-1]
        y = y[:, :-1]
        mean = mean[:, :, :-1]
        logvar = logvar[:, :, :-1]
        max_logvar = max_logvar[:, :-1]
        min_logvar = min_logvar[:, :-1]

    if i_network > -1:
        mean = mean[i_network]
        logvar = logvar[i_network]
        max_logvar = max_logvar[i_network]
        min_logvar = min_logvar[i_network]

    inv_var = torch.exp(-logvar)

    if only_mse:
        return torch.square(mean - y).mean()
    elif only_var_loss:
        return logvar.mean()
    else:
        mse_loss = torch.square(mean - y)
        mse_inv_var_loss = (mse_loss * inv_var).mean()
        var_loss = logvar.mean()

        model.max_logvar += (torch.log(mean.var(dim=1)) - model.max_logvar)*0.001

        if debug:
            print("LR: {:.5f}, MSE: {:.5f}, MSE + INV VAR: {:.5f} VAR: {:.5f}, MIN REW STD: {:.5f}, MAX REW STD: {:.5f}".format(
                model.optim.param_groups[0]['lr'],
                mse_loss.mean().item(),
                mse_inv_var_loss.item(),
                var_loss.item(),
                torch.exp(0.5*max_logvar[:, -1].min()).item(),
                torch.exp(0.5*max_logvar[:, -1].max()).item(),
            ), end='\r')

        return mse_inv_var_loss + var_loss
