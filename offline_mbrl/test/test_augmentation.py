import numpy as np
import pytest
import torch
from offline_mbrl.utils.augmentation import antmaze_augmentation
from offline_mbrl.utils.mazes import ANTMAZE_MEDIUM_MAX, ANTMAZE_MEDIUM_MIN
from offline_mbrl.utils.replay_buffer import ReplayBuffer


@pytest.mark.fast
def test_antmaze_augmentation():
    n_samples = 1000
    obs_dim = 5
    act_dim = 2
    buffer = ReplayBuffer(obs_dim, act_dim, n_samples)

    for _ in range(n_samples):
        buffer.store_batch(
            torch.rand((n_samples, obs_dim)) * 2000 - 1000,
            torch.rand((n_samples, act_dim)) * 2000 - 1000,
            torch.rand((n_samples,)) * 2000 - 1000,
            torch.rand((n_samples, obs_dim)) * 2000 - 1000,
            torch.rand((n_samples,)) * 2000 - 1000,
        )

    batch = buffer.sample_batch(300)

    antmaze_augmentation(
        batch["obs"],
        batch["obs2"],
        ANTMAZE_MEDIUM_MIN,
        ANTMAZE_MEDIUM_MAX,
        ANTMAZE_MEDIUM_MIN,
        ANTMAZE_MEDIUM_MAX,
    )

    for i in range(len(batch["obs"])):
        assert batch["obs"][i][0] < ANTMAZE_MEDIUM_MAX
        assert batch["obs"][i][0] > ANTMAZE_MEDIUM_MIN
        assert batch["obs"][i][1] < ANTMAZE_MEDIUM_MAX
        assert batch["obs"][i][1] > ANTMAZE_MEDIUM_MIN
