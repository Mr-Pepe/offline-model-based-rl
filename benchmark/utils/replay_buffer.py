import torch
from benchmark.utils.combined_shape import combined_shape
import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Based on https://spinningup.openai.com
    """

    def __init__(self, obs_dim, act_dim, size, device='cpu'):
        self.obs_buf = torch.zeros(combined_shape(
            size, obs_dim), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros(combined_shape(
            size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(combined_shape(
            size, act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.bool, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.split_at_size = -1
        self.split_with_val_split = -1
        self.train_idxs = []
        self.val_idxs = []

        self.possible_idxs = None
        self.has_changed = False

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        self.has_changed = True

    def store_batch(self, obs, act, rew, next_obs, done):
        remaining_batch_size = len(obs)
        remaining_buffer_size = self.max_size - self.ptr

        if remaining_batch_size > self.max_size:
            raise ValueError("Batch of size {} does not fit in replay buffer \
                of size {}.".format(remaining_batch_size, self.max_size))

        while remaining_buffer_size < remaining_batch_size:
            self.obs_buf[self.ptr:] = \
                obs[-remaining_batch_size:-
                    remaining_batch_size+remaining_buffer_size]
            self.obs2_buf[self.ptr:] = \
                next_obs[-remaining_batch_size:-
                         remaining_batch_size+remaining_buffer_size]
            self.act_buf[self.ptr:] = \
                act[-remaining_batch_size:-
                    remaining_batch_size+remaining_buffer_size]
            self.rew_buf[self.ptr:] = \
                rew[-remaining_batch_size:-
                    remaining_batch_size+remaining_buffer_size]
            self.done_buf[self.ptr:] = \
                done[-remaining_batch_size:-
                     remaining_batch_size+remaining_buffer_size]

            self.ptr = 0
            self.size = self.max_size

            remaining_batch_size -= remaining_buffer_size
            remaining_buffer_size = self.max_size

        self.obs_buf[self.ptr:self.ptr +
                     remaining_batch_size] = obs[-remaining_batch_size:]
        self.obs2_buf[self.ptr:self.ptr +
                      remaining_batch_size] = next_obs[-remaining_batch_size:]
        self.act_buf[self.ptr:self.ptr +
                     remaining_batch_size] = act[-remaining_batch_size:]
        self.rew_buf[self.ptr:self.ptr +
                     remaining_batch_size] = rew[-remaining_batch_size:]
        self.done_buf[self.ptr:self.ptr +
                      remaining_batch_size] = done[-remaining_batch_size:]

        self.ptr += remaining_batch_size
        self.size = min(self.size+remaining_batch_size, self.max_size)

        self.has_changed = True

    def sample_batch(self, batch_size=32, non_terminals_only=False):
        if self.possible_idxs is not None and not self.has_changed:
            possible_idxs = self.possible_idxs
        else:
            if non_terminals_only and self.done_buf.sum() < self.size:
                possible_idxs = torch.nonzero(
                    (self.done_buf == 0), as_tuple=False)
            else:
                possible_idxs = torch.arange(self.size)

            self.possible_idxs = possible_idxs
            self.has_changed = False

        idxs = possible_idxs[torch.randint(
            0, possible_idxs.numel(), (batch_size,))].flatten()

        obs = self.obs_buf[idxs]
        obs2 = self.obs2_buf[idxs]
        act = self.act_buf[idxs]
        rew = self.rew_buf[idxs]
        done = self.done_buf[idxs]

        batch = dict(obs=obs,
                     obs2=obs2,
                     act=act,
                     rew=rew,
                     done=done)

        return batch

    def sample_train_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or \
                self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.train_idxs, batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_val_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or \
                self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.val_idxs, batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def split(self, val_split=0.2):
        self.split_at_size = self.size
        self.split_with_val_split = val_split

        self.val_idxs = np.random.choice(
            np.arange(self.size), int(self.size*val_split), replace=False)
        self.train_idxs = np.setdiff1d(np.arange(self.size), self.val_idxs)
        self.done_idx = torch.nonzero(self.done_buf, as_tuple=False).view(-1)
        self.not_done_idx = torch.nonzero((self.done_buf == 0),
                                          as_tuple=False).reshape((-1))

    def get_terminal_ratio(self):
        return float(self.done_buf.sum())/self.size

    def has_terminal_state(self):
        return self.done_buf.sum() > 0

    def sample_balanced_terminal_batch(self, batch_size=32, val_split=0.2):
        if not self.has_terminal_state():
            raise Exception("Buffer has no terminal state.")

        if self.split_at_size != self.size or \
                self.split_with_val_split != val_split:
            self.split(val_split)

        done_idxs = np.random.choice(self.done_idx, batch_size//2)
        not_done_idxs = np.random.choice(self.not_done_idx, batch_size//2)

        return dict(obs=torch.cat((self.obs_buf[done_idxs],
                                   self.obs_buf[not_done_idxs])),
                    obs2=torch.cat((self.obs2_buf[done_idxs],
                                    self.obs2_buf[not_done_idxs])),
                    act=torch.cat((self.act_buf[done_idxs],
                                   self.act_buf[not_done_idxs])),
                    rew=torch.cat((self.rew_buf[done_idxs],
                                   self.rew_buf[not_done_idxs])),
                    done=torch.cat((self.done_buf[done_idxs],
                                    self.done_buf[not_done_idxs])))

    def clear(self):
        self.__init__(self.obs_dim,
                      self.act_dim,
                      self.max_size,
                      device=self.device)

    def to(self, device):
        self.obs_buf = self.obs_buf.to(device)
        self.act_buf = self.act_buf.to(device)
        self.obs2_buf = self.obs2_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.done_buf = self.done_buf.to(device)

    def set_curriculum(self, selector):
        self.possible_idxs = selector.select(self)
