# pylint: disable=unused-import
import gym
import torch

from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.termination_functions import get_termination_function

if __name__ == "__main__":
    prefix = "hopper-"
    version = "-v2"

    dataset_names = [
        # 'random',
        "medium-replay",
        # 'medium',
        # 'medium-expert',
        # 'expert',
    ]

    avg_rew = []
    per_trajectory_rews = []

    latex = ""

    for dataset_name in dataset_names:
        env_name = prefix + dataset_name + version
        print(env_name)
        termination_function = get_termination_function(env_name)
        pre_fn = get_preprocessing_function(env_name)

        env = gym.make(env_name)

        buffer, obs_dim, act_dim = load_dataset_from_env(env)

        done_errors = 0
        rew_errors = 0
        obs_errors = 0

        for i in range(10000):  # range(buffer.size):
            if i % 100 == 0:
                print(f"{i}/{buffer.size}", end="\r")

            obs = buffer.obs_buf[i]
            next_obs = buffer.next_obs_buf[i]
            act = buffer.act_buf[i]
            done = buffer.done_buf[i]
            rew = buffer.rew_buf[i]

            env.reset()
            env.set_state(
                torch.cat(
                    (torch.as_tensor([0]), torch.as_tensor(obs[: env.model.nq - 1]))
                ),
                obs[env.model.nq - 1 :],
            )

            real_next_obs, real_rew, _, _ = env.step(act.numpy())

            if abs(real_rew - rew) > 0.01:
                rew_errors += 1

            obs_diff = abs(pre_fn(torch.as_tensor(real_next_obs)) - pre_fn(next_obs))

            deviations = torch.nonzero(obs_diff > 0.01)
            if deviations.numel() > 0:
                obs_errors += 1

        post_dones = termination_function(
            torch.as_tensor(buffer.next_obs_buf).unsqueeze(0)
        )["dones"]

        done_errors = (buffer.done_buf ^ post_dones.view(-1)).sum()

        print(
            f"Obs errors: {obs_errors}  "
            f"Rew errors: {rew_errors}  "
            f"Done errors: {done_errors}"
        )
