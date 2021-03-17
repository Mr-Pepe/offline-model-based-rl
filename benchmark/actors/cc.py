from benchmark.utils.modes import UNDERESTIMATION
from benchmark.models.mlp import mlp
import itertools

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from benchmark.models.mlp_q_function import MLPQFunction
from benchmark.models.squashed_gaussian_mlp_actor import \
    SquashedGaussianMLPActor


class CopyCat(nn.Module):
    def __init__(self, observation_space, action_space, hidden=(128, 128, 128, 128),
                 activation=nn.ReLU, lr=3e-4, batch_size=100, gamma=0.99,
                 pre_fn=None, device='cpu', decay=0.9999, **_):

        super().__init__()

        self.gamma = gamma

        self.batch_size = batch_size
        self.pre_fn = pre_fn
        self.device = device

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # TODO: Action limit for clamping: critically, assumes all dimensions
        # share the same bound!
        act_limit = action_space.high[0]

        self.q = MLPQFunction(obs_dim, act_dim, hidden, activation)
        self.q_optimizer = AdamW(self.q.parameters(), lr=lr)

        self.pi = mlp([obs_dim] + list(hidden) + [act_dim], activation)
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=lr)

        self.to(device)

        self.use_amp = 'cuda' in next(self.parameters()).device.type
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp,
                                                growth_factor=1.5,
                                                backoff_factor=0.7)

        self.k = 3
        self.knn = None
        self.epsilon = 0.5
        self.decay = decay

        self.target = deepcopy(self)

        # Freeze target networks with respect to optimizers (only update via
        # polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_loss_q(self, data):
        o, a, r, o2, d, a2 = data['obs'], data['act'], \
            data['rew'], data['obs2'],  data['done'].bool(), data['act2']

        if self.pre_fn:
            o = self.pre_fn(o)
            o2 = self.pre_fn(o2)

        q = self.q(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target Q-values
            q_pi_targ = self.target.q(o2, a2)
            backup = r + self.gamma * \
                ~d * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        q_info = dict(Q1Vals=q.cpu().detach().numpy(),
                      Q2Vals=0)

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        a = data['act']

        if self.pre_fn:
            o = self.pre_fn(o)

        criterion = nn.MSELoss()

        pi = self.pi(o)

        loss_pi = criterion(pi, a)

        return loss_pi

    def update(self, data):
        self.device = next(self.parameters()).device

        for key in data:
            data[key] = data[key].to(self.device)

        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        self.pi_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_pi = self.compute_loss_pi(data)

        self.scaler.scale(loss_pi).backward()
        self.scaler.step(self.pi_optimizer)
        self.scaler.update()

        return loss_q, q_info, loss_pi

    def multi_update(self, n_updates, buffer, model, logger=None, debug=False):
        self.epsilon *= self.decay

        if self.knn is None:
            self.knn = buffer.get_knn(k=self.k, pre_fn=self.pre_fn, verbose=True)

        losses = torch.zeros(n_updates)
        for i_update in range(n_updates):
            self.eval()
            batch, idxs = buffer.sample_batch(
                self.batch_size, return_idxs=True)

            batch['act'] = self.eps_greedy_actions(batch['obs'], buffer, idxs)

            pred = model.get_prediction(
                torch.cat((batch['obs'], batch['act']), dim=1), mode=UNDERESTIMATION)

            batch['obs2'] = pred[:, :-2]
            batch['rew'] = pred[:, -2]
            batch['done'] = pred[:, -1]

            obs2_knn = buffer.get_knn(
                k=3, pre_fn=self.pre_fn, query=batch['obs2'])
            batch['act2'] = self.eps_greedy_actions(
                batch['obs2'], buffer, range(len(obs2_knn)), obs2_knn)

            self.train()

            loss_q, q_info, loss_pi = self.update(data=batch)

            if logger is not None:
                logger.store(LossQ=loss_q.item(), **q_info)
                logger.store(LossPi=loss_pi.item(), LogPi=0)

        if debug:
            return losses.mean()

    def act(self, o, deterministic=False):
        self.device = next(self.parameters()).device

        obs = torch.as_tensor(o,
                              dtype=torch.float32,
                              device=self.device)

        if self.pre_fn:
            obs = self.pre_fn(obs)

        with torch.no_grad():
            return self.pi(obs)

    def act_randomly(self, o, deterministic=False):
        a = torch.as_tensor([self.action_space.sample() for _ in range(len(o))],
                            device=o.device)
        return a

    def eps_greedy_actions(self, obs, buffer, idxs, knn=None):
        if knn is None:
            knn = self.knn
        act_idx = torch.randint(0, self.k, (self.batch_size,))
        actions = torch.zeros((len(obs), buffer.act_buf.shape[1])).to(obs.device)

        for i_idx, idx in enumerate(idxs):
            if torch.rand((1,)) < self.epsilon:
                actions[i_idx] = buffer.act_buf[int(
                    knn[idx][act_idx[i_idx]])]
            else:
                q_values = self.q(obs[i_idx].unsqueeze(0).repeat(
                    self.k, 1), buffer.act_buf[knn[idx].long()])
                actions[i_idx] = buffer.act_buf[int(
                    knn[idx][q_values.max(dim=0).indices])]

        return actions
