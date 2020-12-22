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
                 pre_fn=None,
                 post_fn=None,
                 rew_fn=None,
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
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.rew_fn = rew_fn

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

        self.optim = None

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Env model parameters: {}".format(n_params))

    def forward(self, raw_obs_act):

        device = next(self.layers.parameters()).device
        raw_obs_act = self.check_device_and_shape(raw_obs_act, device)

        if self.pre_fn:
            obs_act = self.pre_fn(raw_obs_act)
        else:
            obs_act = raw_obs_act.detach().clone()

        pred_obs_deltas, pred_rewards = self.layers(obs_act)

        pred_next_obs = pred_obs_deltas[:, :, :self.obs_dim] + \
            raw_obs_act[:, :self.obs_dim]

        means = torch.cat((pred_next_obs,
                           pred_rewards[:, :, 0].view((self.n_networks, -1, 1))),
                          dim=2)

        if self.type == 'deterministic':

            predictions = means
            logvars = 0

        else:
            next_obs_logvars = pred_next_obs[:, :, self.obs_dim:]
            reward_logvars = pred_rewards[:, :, 1].view(
                (self.n_networks, -1, 1))
            logvars = torch.cat((next_obs_logvars, reward_logvars), dim=2)

            max_logvar = self.max_logvar.unsqueeze(1)
            min_logvar = self.min_logvar.unsqueeze(1)
            logvars = max_logvar - softplus(max_logvar - logvars)
            logvars = min_logvar + softplus(logvars - min_logvar)

            std = torch.exp(0.5*logvars)

            predictions = torch.normal(means, std)

        return predictions, \
            means, \
            logvars, \
            self.max_logvar, \
            self.min_logvar

    def get_prediction(self, raw_obs_act, i_network=-1,
                       pessimism=0, exploration_mode='state',
                       uncertainty='epistemic'):

        device = next(self.layers.parameters()).device

        if i_network == -1:
            i_network = torch.randint(self.n_networks, (1,)).item()

        self.eval()

        self.check_prediction_arguments(uncertainty, pessimism, exploration_mode)

        with torch.no_grad():
            predictions, means, logvars, _, _ = \
                self.forward(raw_obs_act)

        pred_next_obs = predictions[:, :, :-1]
        pred_rewards = predictions[:, :, -1].unsqueeze(-1)

        dones = None

        if self.post_fn:
            post = self.post_fn(
                next_obs=pred_next_obs,
                means=means,
                logvars=logvars)

            if 'dones' in post:
                dones = post['dones'].to(device)
            if 'means' in post:
                means = post['means'].to(device)
            if 'logvars' in post:
                logvars = post['logvars'].to(device)

        if self.rew_fn:
            pred_rewards = self.rew_fn(
                obs=raw_obs_act[:, :self.obs_dim],
                act=raw_obs_act[:, self.obs_dim:],
                next_obs=pred_next_obs)

        if dones is None:
            dones = torch.zeros(
                (self.n_networks, raw_obs_act.shape[0], 1), device=device)

        predictions = torch.cat((pred_next_obs, pred_rewards, dones), dim=2)
        prediction = predictions[i_network]

        if pessimism != 0:

            if exploration_mode == 'reward':
                if uncertainty == 'epistemic':
                    prediction[:, -2] -= pessimism * \
                        means[:, :, -1].std(dim=0).mean(dim=1)
                elif uncertainty == 'aleatoric':
                    prediction[:, -2] -= pessimism * \
                        torch.exp(
                            logvars[:, :, -1]).max(dim=0).values.to(device)

            elif exploration_mode == 'state':
                if uncertainty == 'epistemic':
                    prediction[:, -2] -= pessimism * \
                        means[:, :, :-1].std(dim=0).mean(dim=1)
                elif uncertainty == 'aleatoric':
                    prediction[:, -2] -= pessimism * \
                        torch.exp(
                            logvars[:, :, :-1]).mean(dim=2).max(dim=0).values.to(device)

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
        use_amp = 'cuda' in device.type
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if self.optim is None:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        self.train()

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

                self.optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if self.type == 'deterministic':
                        loss = deterministic_loss(x, y, self)
                    else:
                        loss = probabilistic_loss(x, y, self, debug=debug)

                avg_train_loss += loss.item()
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(self.optim)
                scaler.update()

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

            if debug:
                print('')

        return avg_val_losses, batches_trained

    def check_prediction_arguments(self, uncertainty, pessimism, exploration_mode):
        if not (uncertainty == 'epistemic' or uncertainty == 'aleatoric'):
            raise ValueError(
                "Unknown uncertainty measure: {}".format(uncertainty))

        if not (exploration_mode == 'state' or exploration_mode == 'reward'):
            raise ValueError(
                        "Unknown exploration mode: {}".format(exploration_mode))

        if pessimism != 0 and uncertainty == 'aleatoric' and \
                self.type == 'deterministic':
            raise ValueError(
                "Can not use aleatoric uncertainty with deterministic ensemble.")
