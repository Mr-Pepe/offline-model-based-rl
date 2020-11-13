from benchmark.utils.actions import Actions
from benchmark.train import Trainer
import pytest
import d4rl  # noqa
import numpy as np
from benchmark.utils.termination_functions import termination_functions


@pytest.mark.medium
def test_replay_buffer_is_initially_empty_for_online_training():
    trainer = Trainer('maze2d-open-v0',
                      pretrain_epochs=0)
    assert trainer.real_replay_buffer.size == 0


@pytest.mark.medium
def test_replay_buffer_is_filled_for_offline_training():
    trainer = Trainer('maze2d-open-v0',
                      pretrain_epochs=1)

    assert trainer.real_replay_buffer.size > 0


@pytest.mark.medium
def test_actions_for_online_model_free_training():
    epochs = 3
    steps_per_epoch = 50
    total_steps = epochs*steps_per_epoch
    init_steps = 100
    random_steps = 50

    trainer = Trainer('maze2d-open-dense-v0',
                      epochs=epochs,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      steps_per_epoch=steps_per_epoch,
                      max_ep_len=30,
                      init_steps=init_steps,
                      random_steps=random_steps
                      )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(test_performances.shape, (epochs, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch+1
        assert performance[1] != 0

    np.testing.assert_array_equal(action_log.shape,
                                  (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.UPDATE_AGENT],
                                  [0]*init_steps + [1]*(total_steps-init_steps))
    np.testing.assert_array_equal(action_log[:, Actions.RANDOM_ACTION],
                                  [1]*random_steps +
                                  [0]*(total_steps-random_steps))
    np.testing.assert_array_equal(action_log[:, Actions.GENERATE_ROLLOUTS],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.INTERACT_WITH_ENV],
                                  [1]*total_steps)


@pytest.mark.medium
def test_actions_for_online_model_based_training():
    epochs = 5
    steps_per_epoch = 100
    total_steps = epochs*steps_per_epoch
    init_steps = 300
    random_steps = 50
    train_model_every = 50

    trainer = Trainer('hopper-random-v0',
                      epochs=epochs,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      model_kwargs=dict(hidden=[32, 32],
                                        batch_size=32,
                                        patience=1),
                      use_model=True,
                      train_model_every=train_model_every,
                      steps_per_epoch=steps_per_epoch,
                      max_ep_len=30,
                      init_steps=init_steps,
                      random_steps=random_steps
                      )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(test_performances.shape, (epochs, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch+1
        assert performance[1] != 0

    np.testing.assert_array_equal(action_log.shape,
                                  (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL],
                                  [0]*(init_steps) + [1] +
                                  ([0]*(train_model_every-1) + [1]) * 3 +
                                  [0]*(total_steps - init_steps -
                                       3*train_model_every - 1))
    np.testing.assert_array_equal(action_log[:, Actions.UPDATE_AGENT],
                                  [0]*init_steps + [1]*(total_steps-init_steps))
    np.testing.assert_array_equal(action_log[:, Actions.RANDOM_ACTION],
                                  [1]*random_steps +
                                  [0]*(total_steps-random_steps))
    np.testing.assert_array_equal(action_log[:, Actions.GENERATE_ROLLOUTS],
                                  [0]*init_steps + [1]*(total_steps-init_steps))
    np.testing.assert_array_equal(action_log[:, Actions.INTERACT_WITH_ENV],
                                  [1]*total_steps)


@pytest.mark.medium
def test_actions_for_offline_model_free_training_with_fine_tuning():
    epochs = 3
    pretrain_epochs = 3
    steps_per_epoch = 50
    total_steps = epochs*steps_per_epoch + pretrain_epochs*steps_per_epoch
    init_steps = 100
    random_steps = 50

    trainer = Trainer('maze2d-open-dense-v0',
                      epochs=epochs,
                      pretrain_epochs=pretrain_epochs,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      steps_per_epoch=steps_per_epoch,
                      max_ep_len=30,
                      init_steps=init_steps,
                      random_steps=random_steps
                      )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(test_performances.shape,
                                  (epochs+pretrain_epochs, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch+1-pretrain_epochs
        assert performance[1] != 0

    np.testing.assert_array_equal(action_log.shape,
                                  (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.UPDATE_AGENT],
                                  [1]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.RANDOM_ACTION],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.GENERATE_ROLLOUTS],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.INTERACT_WITH_ENV],
                                  [0]*steps_per_epoch*pretrain_epochs +
                                  [1]*steps_per_epoch*epochs)


@pytest.mark.medium
def test_actions_for_offline_model_based_training_with_fine_tuning():
    epochs = 5
    pretrain_epochs = 3
    steps_per_epoch = 100
    total_steps = epochs*steps_per_epoch + pretrain_epochs*steps_per_epoch
    init_steps = 300
    random_steps = 50
    train_model_every = 50

    trainer = Trainer('hopper-random-v0',
                      epochs=epochs,
                      pretrain_epochs=pretrain_epochs,
                      sac_kwargs=dict(hidden=[32, 32, 32],
                                      batch_size=32),
                      model_kwargs=dict(hidden=[32, 32],
                                        batch_size=32,
                                        patience=0),
                      use_model=True,
                      train_model_every=train_model_every,
                      steps_per_epoch=steps_per_epoch,
                      max_ep_len=30,
                      init_steps=init_steps,
                      random_steps=random_steps,
                      rollouts_per_step=1,
                      )

    test_performances, action_log = trainer.train()

    np.testing.assert_array_equal(test_performances.shape,
                                  (epochs+pretrain_epochs, 2))

    for epoch, performance in enumerate(test_performances):
        assert performance[0] == epoch+1-pretrain_epochs
        assert performance[1] != 0

    np.testing.assert_array_equal(action_log.shape,
                                  (total_steps, len(Actions)))

    np.testing.assert_array_equal(action_log[:, Actions.TRAIN_MODEL],
                                  [1] +
                                  [0]*(pretrain_epochs*steps_per_epoch) +
                                  ([0]*(train_model_every-1) + [1]) * 9 +
                                  [0]*(train_model_every-1))
    np.testing.assert_array_equal(action_log[:, Actions.UPDATE_AGENT],
                                  [1]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.RANDOM_ACTION],
                                  [0]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.GENERATE_ROLLOUTS],
                                  [1]*total_steps)
    np.testing.assert_array_equal(action_log[:, Actions.INTERACT_WITH_ENV],
                                  [0]*steps_per_epoch*pretrain_epochs +
                                  [1]*steps_per_epoch*epochs)


@pytest.mark.medium
def test_throws_error_if_using_model_but_no_termination_fn_available():
    with pytest.raises(ValueError):
        Trainer('maze2d-open-dense-v0', use_model=True)


@pytest.mark.medium
def test_trainer_picks_correct_termination_functions():
    trainer = Trainer('Hopper-v2', use_model=True)

    assert trainer.term_fn == termination_functions['hopper']

    trainer = Trainer('HalfCheetah-v2', use_model=True)

    assert trainer.term_fn == termination_functions['half_cheetah']

    trainer = Trainer('Walker2d-v2', use_model=True)

    assert trainer.term_fn == termination_functions['walker2d']