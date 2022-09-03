#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains a replay buffer ased on https://spinningup.openai.com."""


from typing import Optional, Union

import numpy as np
import torch


class ReplayBuffer:
    """A simple FIFO experience replay buffer.

    Observations and actions are expected to be flattened.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, device: str = "cpu"):
        """Initializes the replay buffer.

        Args:
            obs_dim (int): The size of a single obersation.
            act_dim (int): The size of a single action.
            size (int): The desired replay buffer size.
            device (str, optional): The device to push the replay buffer to. Defaults
                to "cpu".
        """
        self.obs_buf = torch.zeros(
            _combined_shape(size, obs_dim), dtype=torch.float32, device=device
        )
        self.next_obs_buf = torch.zeros(
            _combined_shape(size, obs_dim), dtype=torch.float32, device=device
        )
        self.act_buf = torch.zeros(
            _combined_shape(size, act_dim), dtype=torch.float32, device=device
        )
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.bool, device=device)
        self.pointer, self.size, self.max_size = 0, 0, size

        self.timeouts: Optional[torch.Tensor] = None

        self.split_at_size = -1
        self.split_with_val_split = -1.0
        self.train_idxs: Optional[torch.Tensor] = None
        self.val_idxs: Optional[torch.Tensor] = None
        self.done_idx: Optional[torch.Tensor] = None
        self.not_done_idx: Optional[torch.Tensor] = None

        self.possible_idxs: Optional[torch.Tensor] = None
        self.has_changed = False

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Stores a single sample in the replay buffer.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            rew (torch.Tensor): Reward.
            next_obs (torch.Tensor): Next observation.
            done (torch.Tensor): Terminal signal.
        """
        self.obs_buf[self.pointer] = obs
        self.next_obs_buf[self.pointer] = next_obs
        self.act_buf[self.pointer] = act
        self.rew_buf[self.pointer] = rew
        self.done_buf[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.has_changed = True

    def store_batch(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Stores a batch of samples to the replay buffer.

        The replay buffer overwrites the oldest experience if it is full.

        Args:
            obs (torch.Tensor): Observations.
            act (torch.Tensor): Actions.
            rew (torch.Tensor): Rewards.
            next_obs (torch.Tensor): Next observations.
            done (torch.Tensor): Terminal signals.
        """
        remaining_batch_size = len(obs)
        remaining_buffer_size = self.max_size - self.pointer

        while remaining_buffer_size < remaining_batch_size:
            self.obs_buf[self.pointer :] = obs[
                -remaining_batch_size : -remaining_batch_size + remaining_buffer_size
            ]
            self.next_obs_buf[self.pointer :] = next_obs[
                -remaining_batch_size : -remaining_batch_size + remaining_buffer_size
            ]
            self.act_buf[self.pointer :] = act[
                -remaining_batch_size : -remaining_batch_size + remaining_buffer_size
            ]
            self.rew_buf[self.pointer :] = rew[
                -remaining_batch_size : -remaining_batch_size + remaining_buffer_size
            ]
            self.done_buf[self.pointer :] = done[
                -remaining_batch_size : -remaining_batch_size + remaining_buffer_size
            ]

            self.pointer = 0
            self.size = self.max_size

            remaining_batch_size -= remaining_buffer_size
            remaining_buffer_size = self.max_size

        self.obs_buf[self.pointer : self.pointer + remaining_batch_size] = obs[
            -remaining_batch_size:
        ]
        self.next_obs_buf[
            self.pointer : self.pointer + remaining_batch_size
        ] = next_obs[-remaining_batch_size:]
        self.act_buf[self.pointer : self.pointer + remaining_batch_size] = act[
            -remaining_batch_size:
        ]
        self.rew_buf[self.pointer : self.pointer + remaining_batch_size] = rew[
            -remaining_batch_size:
        ]
        self.done_buf[self.pointer : self.pointer + remaining_batch_size] = done[
            -remaining_batch_size:
        ]

        self.pointer += remaining_batch_size
        self.size = min(self.size + remaining_batch_size, self.max_size)

        self.has_changed = True

    def sample_batch(
        self, batch_size: int = 32, non_terminals_only: bool = False
    ) -> dict[str, torch.Tensor]:
        """Samples a random batch from the replay buffer.

        Args:
            batch_size (int, optional): The batch size. Defaults to 32.
            non_terminals_only (bool, optional): Whether or not to only return samples
                with non-terminal next observations. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the batch.
        """
        if self.possible_idxs is not None and not self.has_changed:
            possible_idxs = self.possible_idxs
        else:
            if non_terminals_only and self.done_buf.sum() < self.size:
                possible_idxs = torch.nonzero((self.done_buf == 0), as_tuple=False)
            else:
                possible_idxs = torch.arange(self.size)

            self.possible_idxs = possible_idxs
            self.has_changed = False

        idxs = (
            possible_idxs[torch.randint(0, possible_idxs.numel(), (batch_size,))]
            .flatten()
            .to(self.obs_buf.device)
        )

        obs = self.obs_buf[idxs]
        next_obs = self.next_obs_buf[idxs]
        act = self.act_buf[idxs]
        rew = self.rew_buf[idxs]
        done = self.done_buf[idxs]

        batch = dict(obs=obs, next_obs=next_obs, act=act, rew=rew, done=done)

        return batch

    def sample_train_batch(
        self, batch_size: int = 32, val_split: float = 0.2
    ) -> dict[str, torch.Tensor]:
        """Samples a training batch from the replay buffer.

        The buffer is split into new training and validation sets if a new
        :code:`val_split` value is passed or if the buffer has not been split yet.

        Args:
            batch_size (int, optional): The batch size. Defaults to 32.
            val_split (float, optional): The relative validation set size. Defaults
                to 0.2.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the batch.
        """
        if self.split_at_size != self.size or self.split_with_val_split != val_split:
            self.split(val_split)

        assert self.train_idxs is not None

        idxs = torch.as_tensor(np.random.choice(self.train_idxs, batch_size))
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )

    def sample_val_batch(
        self, batch_size: int = 32, val_split: float = 0.2
    ) -> dict[str, torch.Tensor]:
        """Samples a validation batch from the replay buffer.

        The buffer is split into new training and validation sets if a new
        :code:`val_split` value is passed or if the buffer has not been split yet.

        Args:
            batch_size (int, optional): The batch size. Defaults to 32.
            val_split (float, optional): The relative validation set size. Defaults
                to 0.2.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the batch.
        """
        if self.split_at_size != self.size or self.split_with_val_split != val_split:
            self.split(val_split)

        assert self.val_idxs is not None

        idxs = torch.as_tensor(np.random.choice(self.val_idxs, batch_size))
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )

    def split(self, val_split: float = 0.2) -> None:
        """Splits the replay buffer into a training and validation set.

        Args:
            val_split (float, optional): The fraction of samples to use for the
                validation set. Defaults to 0.2.
        """
        self.split_at_size = self.size
        self.split_with_val_split = val_split

        self.val_idxs = torch.as_tensor(
            np.random.choice(
                np.arange(self.size), int(self.size * val_split), replace=False
            )
        )
        self.train_idxs = torch.as_tensor(
            np.setdiff1d(np.arange(self.size), self.val_idxs)
        )
        self.done_idx = torch.nonzero(self.done_buf, as_tuple=False).view(-1)
        self.not_done_idx = torch.nonzero((self.done_buf == 0), as_tuple=False).reshape(
            (-1)
        )

    def get_terminal_ratio(self) -> float:
        """Returns the ratio of terminal states to the full buffer size.

        Returns:
            float: The fraction of terminal states in the buffer.
        """
        return float(self.done_buf.sum()) / self.size

    def has_terminal_state(self) -> bool:
        """Determines whether there are any terminal states in the replay buffer.

        Returns:
            bool: Whether there are terminal states in the buffer.
        """
        return bool(self.done_buf.sum() > 0)

    def clear(self) -> None:
        """Clears the replay buffer."""
        # pylint: disable=unnecessary-dunder-call
        self.__init__(self.obs_dim, self.act_dim, self.max_size, device=self.device)  # type: ignore

    def to(self, device: str) -> None:
        """Moves the replay buffer to a device.

        Args:
            device (str): The device to move the replay buffer to.
        """
        self.obs_buf = self.obs_buf.to(device)
        self.act_buf = self.act_buf.to(device)
        self.next_obs_buf = self.next_obs_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.done_buf = self.done_buf.to(device)


def _combined_shape(
    length: int, shape: Optional[Union[int, tuple[int, ...]]] = None
) -> tuple[int, ...]:
    # Based on https://spinningup.openai.com

    if shape is None:
        return (length,)

    return (length, shape) if isinstance(shape, int) else (length, *shape)
