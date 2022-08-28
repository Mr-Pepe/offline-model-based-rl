import os

import d4rl  # pylint: disable=unused-import
import gym
import matplotlib.pyplot as plt
import torch

from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env

if __name__ == "__main__":
    name = "Walker2d"
    prefix = name.lower() + "-"
    version = "-v2"

    dataset_names = [
        "expert",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    device = "cuda"
    env_name = "hopper-expert-v2"
    print(env_name)
    env = gym.make(env_name)

    n_samples = 10000

    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, n_samples, buffer_device=device
    )

    model = torch.load(os.path.join(MODELS_DIR, env_name + "-model.pt"))

    random_samples = 6000
    i_sample = 0
    batch_size = 256

    obs_buf = buffer.obs_buf
    act_buf = buffer.act_buf
    obs2_buf = buffer.obs2_buf

    model_errors = []
    raw_aleatoric_uncertainties = []
    raw_epistemic_uncertainties = []

    while i_sample < len(obs_buf):
        print(i_sample)
        i_to = min(len(obs_buf), i_sample + batch_size)
        obs_act = torch.cat((obs_buf[i_sample:i_to], act_buf[i_sample:i_to]), dim=1)
        obs2 = obs2_buf[i_sample:i_to]

        (
            prediction,
            means,
            logvars,
            explicit_uncertainty,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            underestimated_reward,
        ) = model.get_prediction(obs_act, debug=True)

        model_errors.extend(
            (prediction[:, :-2] - obs2)
            .abs()
            .mean(dim=1)
            .cpu()
            .detach()
            .float()
            .tolist()
        )
        raw_aleatoric_uncertainties.extend(aleatoric_uncertainty.tolist())
        raw_epistemic_uncertainties.extend(epistemic_uncertainty.tolist())

        i_sample += batch_size

    if random_samples > 0:
        print("Generating random samples.")
        env.reset()
        agent = RandomAgent(env, device=device)

        for i_obs in range(random_samples):
            obs = obs_buf[i_obs]
            env.set_state(
                torch.cat(
                    (
                        torch.as_tensor([0]),
                        torch.as_tensor(obs[: env.model.nq - 1].cpu()),
                    )
                ),
                obs[env.model.nq - 1 :].cpu(),
            )
            act = agent.act()

            obs2, _, _, _ = env.step(act.cpu().detach().numpy())

            obs_act = torch.cat((obs.unsqueeze(0), act), dim=1)

            (
                prediction,
                means,
                logvars,
                explicit_uncertainty,
                epistemic_uncertainty,
                aleatoric_uncertainty,
                underestimated_reward,
            ) = model.get_prediction(obs_act, debug=True)

            model_errors.extend(
                (prediction[:, :-2].cpu() - obs2)
                .abs()
                .mean(dim=1)
                .cpu()
                .detach()
                .float()
                .tolist()
            )
            raw_aleatoric_uncertainties.extend(aleatoric_uncertainty.tolist())
            raw_epistemic_uncertainties.extend(epistemic_uncertainty.tolist())

    raw_aleatoric_uncertainties = torch.as_tensor(raw_aleatoric_uncertainties)
    aleatoric_uncertainties = (
        raw_aleatoric_uncertainties / raw_aleatoric_uncertainties.max()
    )
    raw_epistemic_uncertainties = torch.as_tensor(raw_epistemic_uncertainties)
    epistemic_uncertainties = (
        raw_epistemic_uncertainties / raw_epistemic_uncertainties.max()
    )
    model_errors = torch.as_tensor(model_errors)
    model_errors /= model_errors.max()

    combined_uncertainties = torch.sqrt(
        (raw_aleatoric_uncertainties + 1).square() - 1
    ) + torch.sqrt((raw_epistemic_uncertainties + 1).square() - 1)
    combined_uncertainties /= combined_uncertainties.max()

    axes[0].fill_between([0, 1], 0.2, 1, color="grey", alpha=0.5)
    axes[0].plot([-1, 2], [-1, 2], color="black", alpha=0.3, linestyle="--")
    axes[0].plot([-1, 2], [0.2, 0.2], color="black", alpha=0.7)
    axes[0].plot([0.2, 0.2], [-1, 2], color="black", alpha=0.7)
    axes[0].scatter(
        model_errors[:n_samples],
        aleatoric_uncertainties[:n_samples],
        s=1,
        alpha=0.3,
        color="blue",
    )
    axes[0].scatter(
        model_errors[n_samples:],
        aleatoric_uncertainties[n_samples:],
        s=1,
        alpha=0.3,
        color="red",
    )
    axes[0].set_ylabel("Aleatoric uncertainty", fontsize=12)

    axes[1].fill_between([0, 1], 0.2, 1, color="grey", alpha=0.5)
    axes[1].plot([-1, 2], [-1, 2], color="black", alpha=0.3, linestyle="--")
    axes[1].plot([-1, 2], [0.2, 0.2], color="black", alpha=0.7)
    axes[1].plot([0.2, 0.2], [-1, 2], color="black", alpha=0.7)
    axes[1].scatter(
        model_errors[:n_samples],
        epistemic_uncertainties[:n_samples],
        s=1,
        alpha=0.3,
        color="blue",
    )
    axes[1].scatter(
        model_errors[n_samples:],
        epistemic_uncertainties[n_samples:],
        s=1,
        alpha=0.3,
        color="red",
    )
    axes[1].set_ylabel("Epistemic uncertainty", fontsize=12)

    for ax in axes.flat:
        ax.grid(b=True, alpha=0.5, linestyle="--")
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel("Model error")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    fig.subplots_adjust(
        top=0.955, bottom=0.11, left=0.07, right=0.975, hspace=0.2, wspace=0.335
    )

    plt.show()
