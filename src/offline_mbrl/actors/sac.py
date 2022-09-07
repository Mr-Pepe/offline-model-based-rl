import itertools
from copy import deepcopy
from typing import Callable, Optional, Type

import torch
from gym import Space
from torch import nn
from torch.optim.adamw import AdamW

from offline_mbrl.models.mlp_q_function import MLPQFunction
from offline_mbrl.models.squashed_gaussian_mlp_actor import SquashedGaussianMLPActor
from offline_mbrl.utils.logx import EpochLogger  # type: ignore
from offline_mbrl.utils.replay_buffer import ReplayBuffer


class SAC(nn.Module):
    # Based on https://spinningup.openai.com
    # pylint: disable=abstract-method
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        hidden: tuple[int, ...] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
        pi_lr: float = 3e-4,
        q_lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        polyak: float = 0.995,
        batch_size: int = 100,
        preprocessing_function: Callable = None,
        device: str = "cpu",
        **_: dict
    ) -> None:
        """
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

        """

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.pre_fn = preprocessing_function
        self.device = device

        assert observation_space.shape is not None
        assert action_space.shape is not None

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.action_space = action_space

        # Assumes all dimensions share the same bound
        act_limit = action_space.high[0]  # type: ignore

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

    def compute_loss_q(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["next_obs"],
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
            backup = r + self.gamma * ~d * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy()
        )

        return loss_q, q_info

    def compute_loss_pi(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        o = data["obs"]

        if self.pre_fn:
            o = self.pre_fn(o)

        pi, logp_pi = self.pi(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict, torch.Tensor, dict]:
        self.device = next(self.parameters()).device.type

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
                # NB: We use an in-place operations "mul_", "add_" to
                # update target params, as opposed to "mul" and "add", which
                # would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q, q_info, loss_pi, pi_info

    def multi_update(
        self, n_updates: int, buffer: ReplayBuffer, logger: Optional[EpochLogger] = None
    ) -> None:
        for _ in range(n_updates):
            batch = buffer.sample_batch(self.batch_size)
            loss_q, q_info, loss_pi, pi_info = self.update(data=batch)

            if logger is not None:
                logger.store(LossQ=loss_q.item(), **q_info)
                logger.store(LossPi=loss_pi.item(), **pi_info)

    def act(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        self.device = next(self.parameters()).device.type

        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=torch.device(self.device)
        )

        if self.pre_fn:
            obs = self.pre_fn(obs)

        with torch.no_grad():
            action, _ = self.pi(obs, deterministic, False)
            return action

    def act_randomly(
        self, observation: torch.Tensor, unused_deterministic: bool = False
    ) -> torch.Tensor:
        action = torch.as_tensor(
            [self.action_space.sample() for _ in range(len(observation))],
            device=observation.device,
        )
        return action
