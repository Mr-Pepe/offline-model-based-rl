import gym
from benchmark.train import Trainer
import pytest
import d4rl  # noqa
import numpy as np


@pytest.mark.medium
def test_replay_buffer_is_initially_empty_for_online_training():
    trainer = Trainer(lambda: gym.make('maze2d-open-v0'),
                      pretrain_epochs=0)
    assert trainer.real_replay_buffer.size == 0


@pytest.mark.medium
def test_replay_buffer_is_filled_for_offline_training():
    trainer = Trainer(lambda: gym.make('maze2d-open-v0'),
                      pretrain_epochs=1)

    assert trainer.real_replay_buffer.size > 0


@pytest.mark.fast
def test_training_returns_test_performance_for_online_training():
    trainer = Trainer(lambda: gym.make('maze2d-open-dense-v0'),
                      epochs=3,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      steps_per_epoch=50,
                      max_ep_len=30,
                      init_steps=100)

    test_performances = trainer.train()

    np.testing.assert_array_equal(test_performances.shape, (3, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch+1
        assert performance[1] != 0


@pytest.mark.fast
def test_trainer_return_test_performances_for_offline_training():
    trainer = Trainer(lambda: gym.make('maze2d-open-dense-v0'),
                      epochs=0,
                      pretrain_epochs=20,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      steps_per_epoch=50,
                      max_ep_len=30,
                      init_steps=100)

    test_performances = trainer.train()

    np.testing.assert_array_equal(test_performances.shape, (20, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch - 20 + 1
        assert performance[1] != 0
