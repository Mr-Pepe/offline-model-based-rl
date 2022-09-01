import d4rl  # pylint: disable=unused-import
import gym

from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.termination_functions import get_termination_function

if __name__ == "__main__":
    prefix = "walker2d-"
    version = "-v2"

    dataset_names = [
        "random",
        "medium-replay",
        "medium",
        "expert",
        "medium-expert",
    ]

    avg_rew = []
    per_trajectory_rews = []

    latex = ""

    for dataset_name in dataset_names:
        env_name = prefix + dataset_name + version
        print(env_name)
        termination_function = get_termination_function(env_name)

        env = gym.make(env_name)

        buffer, obs_dim, act_dim = load_dataset_from_env(env)

        avg_rew.append(buffer.rew_buf.mean())

        n_timeouts = buffer.timeouts.sum() if buffer.timeouts is not None else 0
        per_trajectory_rew = buffer.rew_buf.sum() / (buffer.done_buf.sum() + n_timeouts)
        per_trajectory_rews.append(per_trajectory_rew)

        latex += (
            f"& {dataset_name} "
            f"& {buffer.size:,} "
            f"& {buffer.rew_buf.mean():.2f} "
            f"& {per_trajectory_rew:.2f} "
            f"& {buffer.done_buf.sum()} "
            f"& {n_timeouts} \\\\ "
        )

    print(latex)

    for i in range(4):
        assert avg_rew[i] < avg_rew[i + 1]
        assert per_trajectory_rews[i] < per_trajectory_rews[i + 1]
