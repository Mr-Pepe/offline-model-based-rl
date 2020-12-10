from benchmark.utils.termination_functions import maze2d_umaze_termination_fn
from benchmark.utils.pretrain_agent import pretrain_agent
import gym
import d4rl  # noqa
from benchmark.actors.sac import SAC
from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.models.environment_model import EnvironmentModel


def agent_pretrains_in_virtual_environment():
    env = gym.make('maze2d-umaze-v1')
    buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples=100000)
    agent = SAC(env.observation_space, env.action_space)
    term_fn = maze2d_umaze_termination_fn
    model = EnvironmentModel(obs_dim, act_dim,
                             term_fn=term_fn)

    model.train_to_convergence(buffer,
                               patience=10)

    n_steps = 1000
    max_rollout_length = 1000
    pessimism = 0
    exploration_mode = 'state'
    n_random_actions = 1000

    pretrain_agent(agent,
                   model,
                   buffer,
                   n_steps=n_steps,
                   n_random_actions=n_random_actions,
                   max_rollout_length=max_rollout_length,
                   pessimism=pessimism,
                   exploration_mode=exploration_mode,
                   debug=True)
