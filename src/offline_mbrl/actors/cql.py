import itertools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from offline_mbrl.models.mlp_q_function import MLPQFunction
from offline_mbrl.models.squashed_gaussian_mlp_actor import SquashedGaussianMLPActor


class CQL(nn.Module):
    # Based on https://spinningup.openai.com

    def __init__(
        self,
        observation_space,
        action_space,
        hidden=(256, 256),
        activation=nn.ReLU,
        pi_lr=3e-4,
        q_lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        batch_size=100,
        n_actions=10,
        pre_fn=None,
        device="cpu",
        **_
    ):

        super().__init__()

        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.pre_fn = pre_fn
        self.device = device

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # TODO: Action limit for clamping: critically, assumes all dimensions
        # share the same bound!
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden, activation, act_limit
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden, activation)

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = AdamW(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = AdamW(self.q_params, lr=q_lr)

        self.to(device)

        self.target = deepcopy(self)

        # Freeze target networks with respect to optimizers (only update via
        # polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

        # CQL
        self.target_entropy = -np.prod(action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = AdamW([self.log_alpha], lr=pi_lr)

        self.n_actions = n_actions
        self.target_action_gap = 10
        self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_prime_optimizer = AdamW([self.log_alpha_prime], lr=q_lr)

        self.temp = 1
        self.min_q_weight = 1

    def compute_loss_q(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        if self.pre_fn:
            o = self.pre_fn(o)
            o2 = self.pre_fn(o2)

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
            backup = r + self.gamma * ~d * (q_pi_targ - self.log_alpha.exp() * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        device = next(self.parameters()).device

        # CQL
        # From https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
        stacked_o = (
            o.unsqueeze(1)
            .repeat(1, self.n_actions, 1)
            .view(o.shape[0] * self.n_actions, o.shape[1])
        )
        stacked_o2 = (
            o2.unsqueeze(1)
            .repeat(1, self.n_actions, 1)
            .view(o.shape[0] * self.n_actions, o.shape[1])
        )
        random_actions = (
            torch.FloatTensor(q1.shape[0] * self.n_actions, a.shape[-1])
            .uniform_(-1, 1)
            .to(device)
        )
        curr_actions, curr_log_pi = self.pi(stacked_o, False, True)
        next_actions, next_log_pi = self.pi(stacked_o2, False, True)
        q1_random = self.q1(stacked_o, random_actions).view(o.shape[0], self.n_actions)
        q2_random = self.q2(stacked_o, random_actions).view(o.shape[0], self.n_actions)
        q1_curr = self.q1(stacked_o, curr_actions).view(o.shape[0], self.n_actions)
        q2_curr = self.q2(stacked_o, curr_actions).view(o.shape[0], self.n_actions)
        q1_next = self.q1(stacked_o, next_actions).view(o.shape[0], self.n_actions)
        q2_next = self.q2(stacked_o, next_actions).view(o.shape[0], self.n_actions)

        random_density = np.log(0.5 ** curr_actions.shape[-1])

        cat_q1 = torch.cat(
            [
                q1_random - random_density,
                q1_next - next_log_pi.detach().view(-1, self.n_actions),
                q1_curr - curr_log_pi.detach().view(-1, self.n_actions),
            ],
            1,
        )
        cat_q2 = torch.cat(
            [
                q2_random - random_density,
                q2_next - next_log_pi.detach().view(-1, self.n_actions),
                q2_curr - curr_log_pi.detach().view(-1, self.n_actions),
            ],
            1,
        )

        min_qf1_loss = (
            torch.logsumexp(
                cat_q1 / self.temp,
                dim=1,
            ).mean()
            * self.min_q_weight
            * self.temp
        )
        min_qf2_loss = (
            torch.logsumexp(
                cat_q2 / self.temp,
                dim=1,
            ).mean()
            * self.min_q_weight
            * self.temp
        )

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2.mean() * self.min_q_weight

        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
        min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
        min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
        alpha_prime_loss.backward(retain_graph=True)
        self.alpha_prime_optimizer.step()

        loss_q = loss_q1 + loss_q2 + min_qf1_loss + min_qf2_loss

        # Useful info for logging
        q_info = dict(Q1Vals=0, Q2Vals=0)

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data["obs"]

        if self.pre_fn:
            o = self.pre_fn(o)

        pi, logp_pi = self.pi(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=0)

        return loss_pi, pi_info

    def update(self, data):
        self.device = next(self.parameters()).device

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

        # Unfreeze Q-networks.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to
                # update target params, as opposed to "mul" and "add", which
                # would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q, q_info, loss_pi, pi_info

    def multi_update(self, n_updates, buffer, logger=None):
        for _ in range(n_updates):
            batch = buffer.sample_batch(self.batch_size)
            loss_q, q_info, loss_pi, pi_info = self.update(data=batch)

            if logger is not None:
                logger.store(LossQ=0, **q_info)
                logger.store(LossPi=0, **pi_info)

    def act(self, o, deterministic=False):
        self.device = next(self.parameters()).device

        obs = torch.as_tensor(o, dtype=torch.float32, device=self.device)

        if self.pre_fn:
            obs = self.pre_fn(obs)

        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def act_randomly(self, o, deterministic=False):
        a = torch.as_tensor(
            [self.action_space.sample() for _ in range(len(o))], device=o.device
        )
        return a
