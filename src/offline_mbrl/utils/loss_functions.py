import torch


def deterministic_loss(x, y, model, i_network=-1):
    y_pred, _, _, _, _, _ = model(x)

    if i_network == -1:
        return torch.square(y_pred - y.unsqueeze(0)).mean()

    return torch.square(y_pred[i_network] - y.unsqueeze(0)).mean()


def probabilistic_loss(
    x,
    y,
    model,
    i_network=-1,
    only_mse=False,
    debug=False,
    no_reward=False,
    only_uncertainty=False,
    pre_fn=None,
):
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
    if only_uncertainty:
        return uncertainty.mean()

    mse_loss = torch.square(mean - y)
    mse_inv_var_loss = (mse_loss * inv_var).mean()
    var_loss = logvar.mean()
    var_bound_loss = 0.01 * max_logvar.mean() - 0.01 * min_logvar.mean()
    uncertainty_loss = uncertainty.mean()

    if debug:
        print(
            f"LR: {model.optim.param_groups[0]['lr']:.5f}, "
            f"State MSE: {mse_loss[:, :, :-1].mean().item():.5f}, "
            f"Rew MSE: {mse_loss[:, :, -1].mean().item():.5f}, "
            f"MSE + INV VAR: {mse_inv_var_loss.item():.5f} "
            f"VAR: {var_loss.item():.5f}, "
            f"BOUNDS: {var_bound_loss.item():.5f}, "
            f"MAX LOGVAR: {max_logvar.mean().item():.5f}, "
            f"MIN LOGVAR: {min_logvar.mean().item():.5f}, ",
            f"UNCERTAINTY: {uncertainty_loss.item():.5f}",
            end="\r",
        )

    return mse_inv_var_loss + var_loss + var_bound_loss + uncertainty_loss
