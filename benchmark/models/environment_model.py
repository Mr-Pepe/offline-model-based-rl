from benchmark.utils.get_x_y_from_batch import get_x_y_from_batch
from benchmark.utils.loss_functions import \
    deterministic_loss, probabilistic_loss
from torch.optim.adam import Adam
from benchmark.models.multi_head_mlp import MultiHeadMlp
import torch.nn as nn
import torch
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class EnvironmentModel(nn.Module):
    """ Takes in a state and action and predicts next state and reward. """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden=[128, 128],
                 type='deterministic',
                 n_networks=1,
                 device='cpu',
                 **_):
        """
            type (string): deterministic or probabilistic

            n_networks (int): number of networks in the ensemble
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Append reward signal
        self.out_dim = obs_dim+1

        self.type = type
        self.n_networks = n_networks

        if type != 'deterministic' and type != 'probabilistic':
            raise ValueError("Unknown type {}".format(type))

        self.layers = MultiHeadMlp(obs_dim, act_dim, hidden, n_networks)

        # Taken from https://github.com/kchua/handful-of-trials/blob/master/
        # dmbrl/modeling/models/BNN.py
        self.max_logvar = Parameter(torch.ones(
            n_networks,
            (self.out_dim))/2,
            requires_grad=True)
        self.min_logvar = Parameter(torch.ones(
            n_networks,
            (self.out_dim))*-10,
            requires_grad=True)

        self.to(device)

    def forward(self, obs_act, term_fn=None):

        device = next(self.layers.parameters()).device
        obs_act = self.check_device_and_shape(obs_act, device)

        next_obs, reward = self.layers(obs_act)

        out = 0
        mean = 0
        logvar = 0
        max_logvar = 0
        min_logvar = 0

        # The model only learns a residual, so the input has to be added
        if self.type == 'deterministic':
            next_obs = next_obs[:, :, :self.obs_dim] + \
                obs_act[:, :self.obs_dim]

            if term_fn:
                done = term_fn(obs=obs_act[:, :self.obs_dim],
                               next_obs=next_obs).to(device)
            else:
                done = torch.zeros((self.n_networks, obs_act.shape[0], 1),
                                   device=device)

            out = torch.cat((next_obs,
                             reward[:, :, 0].view((self.n_networks, -1, 1)),
                             done), dim=2)

        elif self.type == 'probabilistic':
            obs_mean = next_obs[:, :, :self.obs_dim] + \
                obs_act[:, :self.obs_dim]
            reward_mean = reward[:, :, 0].view((self.n_networks, -1, 1))
            mean = torch.cat((obs_mean, reward_mean), dim=2)

            obs_logvar = next_obs[:, :, self.obs_dim:]
            reward_logvar = reward[:, :, 1].view((self.n_networks, -1, 1))
            logvar = torch.cat((obs_logvar, reward_logvar), dim=2)

            max_logvar = self.max_logvar.unsqueeze(1)
            min_logvar = self.min_logvar.unsqueeze(1)
            logvar = max_logvar - softplus(max_logvar - logvar)
            logvar = min_logvar + softplus(logvar - min_logvar)

            std = torch.exp(0.5*logvar)

            out = torch.normal(mean, std)

            if term_fn:
                done = term_fn(obs=obs_act[:, :self.obs_dim],
                               next_obs=out[:, :, :-1]).to(device)

            else:
                done = torch.zeros((self.n_networks, obs_act.shape[0], 1),
                                   device=device)

            out = torch.cat((out, done), dim=2)

        return out, \
            mean, \
            logvar, \
            self.max_logvar, \
            self.min_logvar

    def get_prediction(self, x, i_network=-1, term_fn=None,
                       pessimism=0, exploration_mode='state'):

        if pessimism == 0:
            i_network = torch.randint(self.n_networks,
                                      (1,)) if i_network == -1 else i_network

            with torch.no_grad():
                predictions, _, _, _, _ = self.forward(x,
                                                       term_fn=term_fn)

            prediction = predictions[i_network].view(x.shape[0], -1)

        else:
            if self.type != 'probabilistic':
                raise ValueError("Can not predict pessimistically because \
                    model is not probabilistic")

            device = next(self.layers.parameters()).device

            with torch.no_grad():
                predictions, means, logvars, _, _ = \
                    self.forward(x,
                                 term_fn=term_fn)

            prediction = predictions.mean(dim=0)

            if pessimism > 0:
                # Penalize the reward as in MOPO
                prediction[:, -2] -= pessimism * \
                    torch.exp(logvars[:, :, -1]).to(device).max(dim=0).values
            else:
                if exploration_mode == 'reward':
                    prediction[:, -2] = -pessimism * \
                        torch.exp(logvars[:, :, -1]).to(device).mean(dim=0)

                elif exploration_mode == 'state':
                    # prediction[:, -2] = -pessimism * \
                    #     torch.exp(logvars[:, :, :-1]
                    #               ).to(device).mean(dim=2).mean(dim=0)
                    prediction[:, -2] = -pessimism * \
                        means.std(dim=0).sum(dim=1)

                else:
                    raise ValueError(
                        "Unknown exploration mode: {}".format(exploration_mode))

        prediction[:, -1] = prediction[:, -1] > 0.5
        return prediction

    def check_device_and_shape(self, x, device):
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x

    def train_to_convergence(self, data, lr=1e-3, batch_size=1024,
                             val_split=0.2, patience=20, patience_value=0,
                             debug=False, max_n_train_batches=-1, **_):

        if type(patience) is list:
            if patience_value > 0 and len(patience) > patience_value:
                patience = patience[patience_value]
            else:
                patience = patience[0]

        n_train_batches = int((data.size * (1-val_split)) // batch_size)
        n_val_batches = int((data.size * val_split) // batch_size)

        if n_train_batches == 0 or n_val_batches == 0:
            raise ValueError(
                "Dataset of size {} not big enough to generate a {} % \
                             validation split with batch size {}."
                .format(data.size,
                        val_split*100,
                        batch_size))

        print('')
        print("Buffer size: {} Train batches per epoch: {} Stopping after {} batches".format(
            data.size,
            n_train_batches,
            max_n_train_batches
        ))

        device = next(self.parameters()).device
        optim = Adam(self.parameters(), lr=lr)

        min_val_loss = 1e10
        n_bad_val_losses = 0
        avg_val_loss = 0
        avg_train_loss = 0

        batches_trained = 0

        stop_training = False

        avg_val_losses = torch.zeros((self.n_networks))

        while n_bad_val_losses < patience:

            avg_train_loss = 0

            for i in range(n_train_batches):
                x, y = get_x_y_from_batch(
                    data.sample_train_batch(batch_size,
                                            val_split),
                    device)

                optim.zero_grad()
                if self.type == 'deterministic':
                    loss = deterministic_loss(x, y, self)
                else:
                    loss = probabilistic_loss(x, y, self)

                avg_train_loss += loss.item()
                loss.backward(retain_graph=True)
                optim.step()

                if max_n_train_batches != -1 and \
                        max_n_train_batches <= batches_trained:
                    stop_training = True
                    break

                batches_trained += 1

            if debug:
                print('')
                print("Train loss: {}".format(
                    avg_train_loss/n_train_batches))

            avg_val_losses = torch.zeros((self.n_networks))

            for i_network in range(self.n_networks):
                for i in range(n_val_batches):
                    x, y = get_x_y_from_batch(
                        data.sample_val_batch(batch_size,
                                              val_split),
                        device)

                    if self.type == 'deterministic':
                        avg_val_losses[i_network] += deterministic_loss(
                            x,
                            y,
                            self,
                            i_network).item()
                    else:
                        avg_val_losses[i_network] += probabilistic_loss(
                            x,
                            y,
                            self,
                            i_network,
                            only_mse=True).item()

                avg_val_losses[i_network] /= n_val_batches

            avg_val_loss = avg_val_losses.mean()

            if avg_val_loss < min_val_loss:
                n_bad_val_losses = 0
                min_val_loss = avg_val_loss
            else:
                n_bad_val_losses += 1

            if stop_training:
                break

            print("Train batches: {} Patience: {}/{} Val losses: {}".format(
                batches_trained,
                n_bad_val_losses,
                patience,
                avg_val_losses.tolist()), end='\r')

        return avg_val_losses, batches_trained
