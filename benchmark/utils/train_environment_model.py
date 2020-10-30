from benchmark.utils.loss_functions import \
    deterministic_loss, probabilistic_loss
import torch
from torch.optim.adam import Adam


def train_environment_model(model, data, lr=1e-2, batch_size=1024,
                            val_split=0.2, patience=20, debug=False):
    """ Train an environment model on a replay buffer until convergence """

    data_size = data.size
    n_train_batches = int((data_size * (1-val_split)) // batch_size)
    n_val_batches = int((data_size * val_split) // batch_size)

    if n_train_batches == 0 or n_val_batches == 0:
        raise ValueError(
            """Dataset not big enough to generate a train/val split with the
            given batch size.""")

    avg_val_losses = [1e10 for i in range(model.n_networks)]

    print('')

    done_factor = 1/data.get_terminal_ratio()

    for i_network, network in enumerate(model.networks):
        print("Training network {}/{}".format(i_network+1, model.n_networks))

        device = next(network.parameters()).device
        optim = Adam(network.parameters(), lr=lr)

        min_val_loss = 1e10
        n_bad_val_losses = 0
        avg_val_loss = 0
        avg_train_loss = 0

        while n_bad_val_losses < patience:

            avg_train_loss = 0

            for i in range(n_train_batches):
                batch = data.sample_train_batch(batch_size, val_split)
                x = torch.cat((batch['obs'], batch['act']), dim=1)
                y = torch.cat((batch['obs2'],
                               batch['rew'].unsqueeze(1),
                               batch['done'].unsqueeze(1)), dim=1)

                x = x.to(device)
                y = y.to(device)

                optim.zero_grad()
                if model.type == 'deterministic':
                    loss = deterministic_loss(
                        x, y, model, i_network, done_factor)
                else:
                    loss = probabilistic_loss(
                        x, y, model, i_network,
                        terminal_loss_factor=done_factor)

                avg_train_loss += loss.item()
                loss.backward(retain_graph=True)
                optim.step()

            if debug:
                print("Network: {}/{} Train loss: {}".format(
                    i_network+1,
                    model.n_networks,
                    avg_train_loss/n_train_batches))

            avg_val_loss = 0
            for i in range(n_val_batches):
                batch = data.sample_val_batch(batch_size, val_split)
                x = torch.cat((batch['obs'], batch['act']), dim=1)
                y = torch.cat((batch['obs2'],
                               batch['rew'].unsqueeze(1),
                               batch['done'].unsqueeze(1)), dim=1)

                x = x.to(device)
                y = y.to(device)

                if model.type == 'deterministic':
                    avg_val_loss += deterministic_loss(x,
                                                       y,
                                                       model,
                                                       i_network).item()
                else:
                    avg_val_loss += probabilistic_loss(x,
                                                       y,
                                                       model,
                                                       i_network,
                                                       only_mse=True).item()

            avg_val_loss /= n_val_batches

            if avg_val_loss < min_val_loss:
                n_bad_val_losses = 0
                min_val_loss = avg_val_loss
            else:
                n_bad_val_losses += 1

            if debug:
                print("Network: {}/{} Patience: {} Val loss: {}".format(
                    i_network+1,
                    model.n_networks,
                    n_bad_val_losses,
                    avg_val_loss))

        avg_val_losses[i_network] = avg_val_loss

    return avg_val_losses
