from benchmark.utils.get_x_y_from_batch import get_x_y_from_batch
from benchmark.utils.loss_functions import \
    deterministic_loss, probabilistic_loss
from torch.optim.adam import Adam
from benchmark.utils.termination_functions import termination_functions
from benchmark.models.mlp import mlp
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
                 n_networks=1):
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

        self.networks = []

        for i_network in range(n_networks):
            self.networks.append(MultiHeadMlp(
                [obs_dim + act_dim] + hidden,
                self.obs_dim,
                double_output=type == 'probabilistic'))

            # Taken from https://github.com/kchua/handful-of-trials/blob/master/
            # dmbrl/modeling/models/BNN.py
            self.networks[i_network].max_logvar = Parameter(torch.ones(
                (self.out_dim))/2,
                requires_grad=True)
            self.networks[i_network].min_logvar = Parameter(torch.ones(
                (self.out_dim))*-10,
                requires_grad=True)

        self.networks = nn.ModuleList(self.networks)

        self.done_network = mlp([self.obs_dim, 32, 32, 32, 1],
                                nn.ReLU,
                                nn.Sigmoid)

    def forward(self, obs_act, i_network=0, term_fn=None):

        if term_fn and term_fn not in termination_functions:
            raise ValueError(
                "Unknown termination function: {}".format(term_fn))

        network = self.networks[i_network]

        device = next(network.parameters()).device
        obs_act = self.check_device_and_shape(obs_act, device)

        # TODO: Transform angular input to [sin(x), cos(x)]

        obs, reward = network(obs_act)

        out = 0
        mean = 0
        logvar = 0
        max_logvar = 0
        min_logvar = 0

        # The model only learns a residual, so the input has to be added
        if self.type == 'deterministic':
            obs = obs + obs_act[:, :self.obs_dim]

            if term_fn:
                done = termination_functions[term_fn](obs)
            else:
                done = self.done_network(obs)

            out = torch.cat((obs, reward, done), dim=1)

        elif self.type == 'probabilistic':
            obs_mean = obs[:, :self.obs_dim] + obs_act[:, :self.obs_dim]
            reward_mean = reward[:, 0].unsqueeze(1)
            mean = torch.cat((obs_mean, reward_mean), dim=1)

            obs_logvar = obs[:, self.obs_dim:]
            reward_logvar = reward[:, 1].unsqueeze(1)
            max_logvar = network.max_logvar
            min_logvar = network.min_logvar

            logvar = torch.cat((obs_logvar, reward_logvar), dim=1)
            logvar = max_logvar - softplus(max_logvar - logvar)
            logvar = min_logvar + softplus(logvar - min_logvar)

            std = torch.exp(0.5*logvar)

            out = torch.normal(mean, std)

            if term_fn:
                done = termination_functions[term_fn](out[:, :-1])
            else:
                done = self.done_network(out[:, :-1])

            out = torch.cat((out, done), dim=1)

        return out, \
            mean, \
            logvar, \
            max_logvar, \
            min_logvar

    def get_prediction(self, x, i_network=-1, term_fn=None):
        i_network = torch.randint(self.n_networks,
                                  (1,)) if i_network == -1 else i_network

        prediction, _, _, _, _ = self.forward(x, i_network, term_fn=term_fn)

        prediction[:, -1] = prediction[:, -1] > 0.5

        return prediction.cpu().detach().numpy()

    def check_device_and_shape(self, x, device):
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x

    def train_to_convergence(self, data, lr=1e-3, batch_size=1024,
                             val_split=0.2, patience=20, debug=False):

        n_train_batches = int((data.size * (1-val_split)) // batch_size)
        n_val_batches = int((data.size * val_split) // batch_size)

        if n_train_batches == 0 or n_val_batches == 0:
            raise ValueError(
                """Dataset not big enough to generate a train/val split with the
                given batch size.""")

        avg_val_losses = [1e10 for i in range(self.n_networks)]

        print('')

        for i_network, network in enumerate(self.networks):
            print("Training network {}/{}".format(i_network+1, self.n_networks))

            device = next(network.parameters()).device
            optim = Adam(network.parameters(), lr=lr)

            min_val_loss = 1e10
            n_bad_val_losses = 0
            avg_val_loss = 0
            avg_train_loss = 0

            while n_bad_val_losses < patience:

                avg_train_loss = 0

                for i in range(n_train_batches):
                    x, y = get_x_y_from_batch(
                        data.sample_train_batch(batch_size,
                                                val_split),
                        device)

                    optim.zero_grad()
                    if self.type == 'deterministic':
                        loss = deterministic_loss(x, y, self, i_network)
                    else:
                        loss = probabilistic_loss(x, y, self, i_network)

                    avg_train_loss += loss.item()
                    loss.backward(retain_graph=True)
                    optim.step()

                if debug:
                    print("Network: {}/{} Train loss: {}".format(
                        i_network+1,
                        self.n_networks,
                        avg_train_loss/n_train_batches))

                avg_val_loss = 0
                for i in range(n_val_batches):
                    x, y = get_x_y_from_batch(
                        data.sample_val_batch(batch_size,
                                              val_split),
                        device)

                    if self.type == 'deterministic':
                        avg_val_loss += deterministic_loss(x,
                                                           y,
                                                           self,
                                                           i_network).item()
                    else:
                        avg_val_loss += probabilistic_loss(x,
                                                           y,
                                                           self,
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
                        self.n_networks,
                        n_bad_val_losses,
                        avg_val_loss))

            avg_val_losses[i_network] = avg_val_loss

        return avg_val_losses
