import torch


def deterministic_loss(x, y, model, i_network=-1):
    y_pred, _, _, _, _, _ = model(x)

    if i_network == -1:
        return torch.square(y_pred - y.unsqueeze(0)).mean()
    else:
        return torch.square(y_pred[i_network] - y.unsqueeze(0)).mean()


def probabilistic_loss(x, y, model, i_network=-1, only_mse=False, debug=False, no_reward=False,
                       only_uncertainty=False, pre_fn=None, augment=False):
    _, mean, logvar, max_logvar, min_logvar, uncertainty = model(x)

    if pre_fn is not None:
        mean[:, :, :-1] = pre_fn(mean[:, :, :-1], detach=False)
        y[:, :-1] = pre_fn(y[:, :-1])

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
    elif only_uncertainty:
        return uncertainty.mean()
    else:
        mse_loss = torch.square(mean - y)
        mse_inv_var_loss = (mse_loss * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.mean() - 0.01 * min_logvar.mean()
        uncertainty_loss = uncertainty.mean()

        epistemic_loss = torch.zeros((1,))
        if augment:
            contiguous_mean = torch.transpose(mean, 0, 1).clone().contiguous()
            epistemic_loss = torch.cdist(
                contiguous_mean,
                contiguous_mean).max(-1).values.max(-1).values.mean()
        if debug:
            print("LR: {:.5f}, State MSE: {:.5f}, Rew MSE: {:.5f}, MSE + INV VAR: {:.5f} VAR: {:.5f}, BOUNDS: {:.5f}, MAX LOGVAR: {:.5f}, MIN LOGVAR: {:.5f}, UNCERTAINTY: {:.5f}, EPISTEMIC: {:.5f}".format(
                model.optim.param_groups[0]['lr'],
                mse_loss[:, :, :-1].mean().item(),
                mse_loss[:, :, -1].mean().item(),
                mse_inv_var_loss.item(),
                var_loss.item(),
                var_bound_loss.item(),
                max_logvar.mean().item(),
                min_logvar.mean().item(),
                uncertainty_loss.item(),
                epistemic_loss.item()
            ), end='\r')

        loss = mse_inv_var_loss + var_loss + var_bound_loss + uncertainty_loss

        if augment:
            loss += epistemic_loss

        return loss
