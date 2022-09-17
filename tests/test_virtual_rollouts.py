import gym
import numpy as np
import pytest
import torch

from offline_mbrl.actors.sac import SAC
from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.schemas import EnvironmentModelConfiguration, SACConfiguration
from offline_mbrl.utils.envs import HALF_CHEETAH_RANDOM_V2, HOPPER_RANDOM_V2
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.termination_functions import get_termination_function
from offline_mbrl.utils.virtual_rollouts import generate_virtual_rollouts


@pytest.mark.fast
def test_generate_rollout_of_desired_length() -> None:
    env = gym.make("HalfCheetah-v2")
    observation_space = env.observation_space
    action_space = env.action_space

    assert observation_space.shape is not None
    assert action_space.shape is not None

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=3)
    buffer.store(
        torch.as_tensor(start_observation),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        termination_function=get_termination_function(HALF_CHEETAH_RANDOM_V2),
    )

    model = EnvironmentModel(obs_dim, act_dim, config=model_config)
    agent = SAC(observation_space, action_space)

    virtual_rollout, _ = generate_virtual_rollouts(
        model, agent, buffer, 10, n_rollouts=1
    )

    assert len(virtual_rollout["obs"]) == 10

    np.testing.assert_array_equal(virtual_rollout["obs"].shape, (10, 17))
    np.testing.assert_array_equal(virtual_rollout["act"].shape, (10, 6))
    np.testing.assert_array_equal(virtual_rollout["rew"].shape, (10))
    np.testing.assert_array_equal(virtual_rollout["next_obs"].shape, (10, 17))
    np.testing.assert_array_equal(virtual_rollout["done"].shape, (10))


@pytest.mark.fast
def test_generate_rollout_stops_on_terminal() -> None:
    torch.manual_seed(0)
    env = gym.make("HalfCheetah-v2")
    observation_space = env.observation_space
    action_space = env.action_space

    assert observation_space.shape is not None
    assert action_space.shape is not None

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=3)
    buffer.store(
        torch.as_tensor(start_observation),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        termination_function=lambda next_obs, **_: torch.randint(2, (1, 1, 1)),
    )

    model = EnvironmentModel(obs_dim, act_dim, config=model_config)
    agent = SAC(observation_space, action_space)

    virtual_rollout, _ = generate_virtual_rollouts(
        model,
        agent,
        buffer,
        10,
    )

    assert len(virtual_rollout["obs"]) < 10


@pytest.mark.medium
def test_use_random_actions_in_virtual_rollout() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(HOPPER_RANDOM_V2)
    observation_space = env.observation_space
    action_space = env.action_space

    assert observation_space.shape is not None
    assert action_space.shape is not None

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    start_observation = torch.as_tensor(
        env.reset(), dtype=torch.float32, device=torch.device(device)
    ).unsqueeze(0)

    buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device
    )
    buffer.store(
        torch.as_tensor(start_observation),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
        torch.as_tensor(0),
    )

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        termination_function=get_termination_function(HOPPER_RANDOM_V2),
    )

    model = EnvironmentModel(obs_dim, act_dim, config=model_config)
    model.to(device)
    agent = SAC(observation_space, action_space, SACConfiguration(device=device))

    torch.random.manual_seed(0)
    rollouts1, _ = generate_virtual_rollouts(model, agent, buffer, 1, 100)
    torch.random.manual_seed(0)
    rollouts2, _ = generate_virtual_rollouts(model, agent, buffer, 1, 100)
    torch.random.manual_seed(0)
    rollouts3, _ = generate_virtual_rollouts(
        model, agent, buffer, 1, 100, random_action=True
    )
    torch.random.manual_seed(0)
    rollouts4, _ = generate_virtual_rollouts(
        model, agent, buffer, 1, 100, random_action=True
    )

    np.testing.assert_array_equal(
        rollouts1["next_obs"].cpu(), rollouts2["next_obs"].cpu()
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        rollouts3["next_obs"].cpu(),
        rollouts4["next_obs"].cpu(),
    )


@pytest.mark.medium
def test_continuously_grow_rollouts() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = HOPPER_RANDOM_V2

    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space

    seed = 1
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    assert observation_space.shape is not None
    assert action_space.shape is not None

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    buffer, _, _ = load_dataset_from_env(env_name, 10000, buffer_device=device)

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        n_networks=3,
        device=device,
        termination_function=get_termination_function(HOPPER_RANDOM_V2),
        training_patience=1,
    )

    model = EnvironmentModel(obs_dim, act_dim, config=model_config)

    model.train_to_convergence(buffer, config=model_config)

    # Reinitialize buffer
    buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device
    )

    virtual_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=int(1e6), device=device
    )

    o = env.reset()

    steps = 1000

    for _ in range(steps):
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

        if d:
            o = env.reset()

    agent = SAC(observation_space, action_space, SACConfiguration(device=device))

    last_observations = None

    steps = 100
    n_rollouts = 100
    steps_per_rollout = 1
    max_rollout_length = 10

    for _ in range(steps):
        rollouts, last_observations = generate_virtual_rollouts(
            model,
            agent,
            buffer,
            steps=steps_per_rollout,
            n_rollouts=n_rollouts,
            random_action=True,
            prev_obs=last_observations,
            max_rollout_length=max_rollout_length,
        )

        virtual_buffer.store_batch(
            rollouts["obs"],
            rollouts["act"],
            rollouts["rew"],
            rollouts["next_obs"],
            rollouts["done"],
        )

        for length in last_observations["lengths"]:
            assert 0 < length <= max_rollout_length
