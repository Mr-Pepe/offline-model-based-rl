import itertools

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from benchmark.models.mlp_q_function import MLPQFunction
from benchmark.models.squashed_gaussian_mlp_actor import \
    SquashedGaussianMLPActor


class SAC(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU, pi_lr=1e-3, q_lr=1e-3, gamma=0.99, alpha=0.2,
                 polyak=0.995, device='cpu'):
        '''
            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

        '''

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.device = device

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # TODO: Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.q1.parameters(), self.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        self.to(device)
        # self.pi.to(device)
        # self.q1.to(device)
        # self.q2.to(device)

        self.target = deepcopy(self)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.pi(o2)

            # Target Q-values
            q1_pi_targ = self.target.q1(o2, a2)
            q2_pi_targ = self.target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.pi(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, data):
        for key in data:
            data[key] = data[key].to(self.device)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q, q_info, loss_pi, pi_info

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()

    def get_action(self, o, deterministic=False):
        return self.act(torch.as_tensor(o, dtype=torch.float32, device=self.device),
                        deterministic)