import torch
from benchmark.utils.combined_shape import combined_shape
import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Based on https://spinningup.openai.com
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.split_at_size = -1
        self.split_with_val_split = -1
        self.train_idxs = []
        self.val_idxs = []

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def sample_train_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.train_idxs, batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def sample_val_batch(self, batch_size=32, val_split=0.2):
        if self.split_at_size != self.size or self.split_with_val_split != val_split:
            self.split(val_split)

        idxs = np.random.choice(self.val_idxs, batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def split(self, val_split=0.2):
        self.split_at_size = self.size
        self.split_with_val_split = val_split

        self.val_idxs = np.random.choice(np.arange(self.size), int(self.size*val_split), replace=False)
        self.train_idxs = np.setdiff1d(np.arange(self.size), self.val_idxs)

    def get_terminal_ratio(self):
        return self.done_buf.sum()/self.size
