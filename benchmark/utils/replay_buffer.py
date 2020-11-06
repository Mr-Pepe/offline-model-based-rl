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
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.split_at_size = -1
        self.split_with_val_split = -1
        self.train_idxs = []
        self.val_idxs = []

        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = torch.as_tensor(obs,
                                                 dtype=torch.float32,
                                                 device=self.device)
        self.obs2_buf[self.ptr] = torch.as_tensor(next_obs,
                                                  dtype=torch.float32,
                                                  device=self.device)
        self.act_buf[self.ptr] = torch.as_tensor(act,
                                                 dtype=torch.float32,
                                                 device=self.device)
        self.rew_buf[self.ptr] = torch.as_tensor(rew,
                                                 dtype=torch.float32,
                                                 device=self.device)
        self.done_buf[self.ptr] = torch.as_tensor(done,
                                                  dtype=torch.float32,
                                                  device=self.device)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def store_batch(self, obs, act, rew, next_obs, done):
        if self.size > 0:
            raise RuntimeError("Batch can only be stored in empty buffer.")

        batch_size = len(obs)

        if self.max_size < batch_size:
            raise ValueError("Buffer not big enough to add batch.")

        print("Pushing batch to GPU if necessary.")
        obs = torch.as_tensor(obs,
                              dtype=torch.float32,
                              device=self.device)
        next_obs = torch.as_tensor(next_obs,
                                   dtype=torch.float32,
                                   device=self.device)
        act = torch.as_tensor(act,
                              dtype=torch.float32,
                              device=self.device)
        rew = torch.as_tensor(rew,
                              dtype=torch.float32,
                              device=self.device)
        done = torch.as_tensor(done,
                               dtype=torch.float32,
                               device=self.device)

        print("Adding batch to buffer.")

        self.obs_buf[:batch_size] = obs
        self.obs2_buf[:batch_size] = next_obs
        self.act_buf[:batch_size] = act
        self.rew_buf[:batch_size] = rew
        self.done_buf[:batch_size] = done

        self.ptr = batch_size
        self.size = batch_size

        # for i in range(len(obs)):
        #     self.store(obs[i],
        #                act[i],
        #                rew[i],
        #                next_obs[i],
        #                done[i])

    def sample_batch(self, batch_size=32):
        idxs = torch.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in batch.items()}

    def sample_train_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or \
                self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.train_idxs, batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in batch.items()}

    def sample_val_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or \
                self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.val_idxs, batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in batch.items()}

    def split(self, val_split=0.2):
        self.split_at_size = self.size
        self.split_with_val_split = val_split

        self.val_idxs = np.random.choice(
            np.arange(self.size), int(self.size*val_split), replace=False)
        self.train_idxs = np.setdiff1d(np.arange(self.size), self.val_idxs)
        self.done_idx = np.argwhere(self.done_buf == True).reshape((-1))
        self.not_done_idx = np.argwhere(self.done_buf == False).reshape((-1))

    def get_terminal_ratio(self):
        return self.done_buf.sum()/self.size

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

        batch = dict(obs=torch.cat((self.obs_buf[done_idxs],
                                   self.obs_buf[not_done_idxs])),
                     obs2=torch.cat((self.obs2_buf[done_idxs],
                                    self.obs2_buf[not_done_idxs])),
                     act=torch.cat((self.act_buf[done_idxs],
                                   self.act_buf[not_done_idxs])),
                     rew=torch.cat((self.rew_buf[done_idxs],
                                   self.rew_buf[not_done_idxs])),
                     done=torch.cat((self.done_buf[done_idxs],
                                    self.done_buf[not_done_idxs])))

        return {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in batch.items()}
