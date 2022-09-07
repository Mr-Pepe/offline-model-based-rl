import d4rl  # pylint: disable=unused-import
import gym
import numpy as np
import pytest
import torch

from offline_mbrl.utils.envs import HOPPER_RANDOM_V2
from offline_mbrl.utils.replay_buffer import ReplayBuffer


@pytest.mark.medium
def test_buffer_returns_percentage_of_terminal_states() -> None:
    env = gym.make(HOPPER_RANDOM_V2)
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    dones = dataset["terminals"]

    buffer = ReplayBuffer(len(observations[0]), len(actions[0]), len(observations))

    buffer.store_batch(
        torch.as_tensor(observations),
        torch.as_tensor(actions),
        torch.as_tensor(rewards),
        torch.as_tensor(next_observations),
        torch.as_tensor(dones),
    )

    np.testing.assert_almost_equal(
        buffer.get_terminal_ratio(), dones.sum() / dones.size
    )


@pytest.mark.medium
def test_add_batches_to_buffer() -> None:
    n_samples = 100000
    size_first_batch = 1234

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(HOPPER_RANDOM_V2)
    dataset = d4rl.qlearning_dataset(env)
    observations = dataset["observations"][:n_samples]
    next_observations = dataset["next_observations"][:n_samples]
    actions = dataset["actions"][:n_samples]
    rewards = dataset["rewards"][:n_samples]
    dones = dataset["terminals"][:n_samples]

    buffer = ReplayBuffer(len(observations[0]), len(actions[0]), 2000000, device=device)

    assert buffer.size == 0

    buffer.store_batch(
        torch.as_tensor(observations)[:size_first_batch],
        torch.as_tensor(actions)[:size_first_batch],
        torch.as_tensor(rewards)[:size_first_batch],
        torch.as_tensor(next_observations)[:size_first_batch],
        torch.as_tensor(dones)[:size_first_batch],
    )

    assert buffer.size == size_first_batch
    np.testing.assert_array_equal(buffer.obs_buf[0].cpu(), observations[0])
    np.testing.assert_array_equal(
        buffer.obs_buf[size_first_batch - 1].cpu(), observations[size_first_batch - 1]
    )

    buffer.store_batch(
        torch.as_tensor(observations)[size_first_batch:],
        torch.as_tensor(actions)[size_first_batch:],
        torch.as_tensor(rewards)[size_first_batch:],
        torch.as_tensor(next_observations)[size_first_batch:],
        torch.as_tensor(dones)[size_first_batch:],
    )

    assert buffer.size == n_samples
    assert buffer.pointer == n_samples

    np.testing.assert_array_equal(buffer.obs_buf[0].cpu(), observations[0])
    np.testing.assert_array_equal(
        buffer.obs_buf[size_first_batch - 1].cpu(), observations[size_first_batch - 1]
    )
    np.testing.assert_array_equal(buffer.obs_buf[n_samples - 1].cpu(), observations[-1])


@pytest.mark.medium
def test_add_experience_to_buffer_online() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("HalfCheetah-v2")
    buffer = ReplayBuffer(
        env.observation_space.shape, env.action_space.shape[0], 150, device=device
    )

    o = env.reset()
    for _ in range(100):
        a = env.action_space.sample()

        o2, r, d, _ = env.step(a)

        buffer.store(
            torch.as_tensor(o),
            torch.as_tensor(a),
            torch.as_tensor(r),
            torch.as_tensor(o2),
            torch.as_tensor(d),
        )

        o = o2

    assert buffer.size == 100


@pytest.mark.fast
def test_buffer_returns_whether_it_contains_a_done_state() -> None:
    buffer = ReplayBuffer(1, 1, 100)

    assert not buffer.has_terminal_state()

    for _ in range(100):
        buffer.store(
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            False,
        )

    assert not buffer.has_terminal_state()

    for _ in range(100):
        buffer.store(
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            True,
        )

    assert buffer.has_terminal_state()


@pytest.mark.medium
def test_store_batch_to_prefilled_buffer_that_is_too_small() -> None:
    n_samples = 137
    buffer_size = 100
    prefill = 80

    env = gym.make(HOPPER_RANDOM_V2)
    dataset = d4rl.qlearning_dataset(env)
    observations = torch.as_tensor(dataset["observations"][:n_samples])
    next_observations = torch.as_tensor(dataset["next_observations"][:n_samples])
    actions = torch.as_tensor(dataset["actions"][:n_samples])
    rewards = torch.as_tensor(dataset["rewards"][:n_samples])
    dones = torch.as_tensor(dataset["terminals"][:n_samples])

    buffer = ReplayBuffer(len(observations[0]), len(actions[0]), buffer_size)

    buffer.store_batch(
        observations[:prefill],
        actions[:prefill],
        rewards[:prefill],
        next_observations[:prefill],
        dones[:prefill],
    )

    buffer.store_batch(
        observations[prefill:],
        actions[prefill:],
        rewards[prefill:],
        next_observations[prefill:],
        dones[prefill:],
    )

    assert buffer.pointer == n_samples % buffer_size
    np.testing.assert_array_equal(buffer.obs_buf[buffer.pointer - 1], observations[-1])


@pytest.mark.fast
def test_clear_buffer() -> None:
    buffer = ReplayBuffer(1, 1, 200)

    for _ in range(100):
        buffer.store(
            torch.as_tensor(1),
            torch.as_tensor(1),
            torch.as_tensor(1),
            torch.as_tensor(1),
            False,
        )

    assert buffer.max_size == 200
    assert buffer.device == "cpu"
    assert buffer.act_dim == 1
    assert buffer.obs_dim == 1
    assert buffer.pointer == 100
    assert buffer.size == 100
    assert buffer.obs_buf[80] == 1

    buffer.clear()

    assert buffer.max_size == 200
    assert buffer.device == "cpu"
    assert buffer.act_dim == 1
    assert buffer.obs_dim == 1
    assert buffer.pointer == 0
    assert buffer.size == 0
    assert buffer.obs_buf[80] == 0


@pytest.mark.fast
def test_buffer_to_device(request):
    if not torch.cuda.is_available():
        mark = pytest.mark.xfail(reason="CUDA not available")
        request.node.add_marker(mark)

    buffer = ReplayBuffer(1, 1, 100)

    for _ in range(100):
        buffer.store(
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            torch.as_tensor(0),
            False,
        )

    buffer.to("cuda")

    assert buffer.obs_buf.device == torch.device("cuda", index=0)
    assert buffer.next_obs_buf.device == torch.device("cuda", index=0)
    assert buffer.act_buf.device == torch.device("cuda", index=0)
    assert buffer.rew_buf.device == torch.device("cuda", index=0)
    assert buffer.done_buf.device == torch.device("cuda", index=0)
