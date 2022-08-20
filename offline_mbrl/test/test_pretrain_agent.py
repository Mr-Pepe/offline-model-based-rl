import d4rl  # noqa
import gym
from offline_mbrl.actors.sac import SAC
from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.postprocessing import postprocess_maze2d_umaze
from offline_mbrl.utils.pretrain_agent import pretrain_agent


def agent_pretrains_in_virtual_environment():
    env = gym.make('maze2d-umaze-v1')
    buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples=100000)
    agent = SAC(env.observation_space, env.action_space)
    post_fn = postprocess_maze2d_umaze
    model = EnvironmentModel(obs_dim, act_dim,
                             post_fn=post_fn)

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
