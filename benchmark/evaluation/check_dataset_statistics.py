from benchmark.utils.load_dataset import load_dataset_from_env
from benchmark.utils.postprocessing import get_postprocessing_function
import gym
import d4rl # noqa

prefix = 'walker2d-'
version = '-v2'

dataset_names = [
    'random',
    'medium-replay',
    'medium',
    'expert',
    'medium-expert',
]

avg_rew = []
per_trajectory_rews = []

latex = ''

for dataset_name in dataset_names:
    env_name = prefix + dataset_name + version
    print(env_name)
    post_fn = get_postprocessing_function(env_name)

    env = gym.make(env_name)

    buffer, obs_dim, act_dim = load_dataset_from_env(env, with_timeouts=True)

    avg_rew.append(buffer.rew_buf.mean())

    n_timeouts = buffer.timeouts.sum() if buffer.timeouts is not None else 0
    per_trajectory_rew = buffer.rew_buf.sum() / (buffer.done_buf.sum() + n_timeouts)
    per_trajectory_rews.append(per_trajectory_rew)

    latex += "& {} & {:,} & {:.2f} & {:.2f} & {} & {} \\\\ ".format(
        dataset_name,
        buffer.size,
        buffer.rew_buf.mean(),
        per_trajectory_rew,
        buffer.done_buf.sum(),
        n_timeouts
    )

print(latex)

for i in range(4):
    assert avg_rew[i] < avg_rew[i+1]
    assert per_trajectory_rews[i] < per_trajectory_rews[i+1]
