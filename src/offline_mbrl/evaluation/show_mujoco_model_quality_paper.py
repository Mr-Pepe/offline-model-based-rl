import os

import d4rl  # pylint: disable=unused-import
import gym
import matplotlib.pyplot as plt
import torch

from offline_mbrl.actors.random_agent import RandomAgent
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env

if __name__ == "__main__":
    name = "Hopper"
    prefix = name.lower() + "-"
    version = "-v2"

    dataset_names = [
        # 'random',
        # 'medium-replay',
        # 'medium',
        # 'medium-expert',
        "expert",
    ]

    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(9, 3.5))

    axes = axes.reshape((-1, 2))

    # fig.suptitle(name, fontsize=24)

    for i_dataset, dataset_name in enumerate(dataset_names):
        device = "cuda"
        env_name = prefix + dataset_name + version
        print(env_name)
        env = gym.make(env_name)

        n_samples = 3000

        buffer, obs_dim, act_dim = load_dataset_from_env(env, n_samples)

        model = torch.load(os.path.join(MODELS_DIR, env_name + "-model.pt"))

        random_samples = 1000
        i_sample = 0
        batch_size = 256

        idxs = torch.randint(0, buffer.size, (n_samples,))
        obs_buf = buffer.obs_buf[idxs].cuda()
        act_buf = buffer.act_buf[idxs].cuda()
        next_obs_buf = buffer.next_obs_buf[idxs].cuda()

        model_errors = []
        raw_aleatoric_uncertainties = []
        raw_epistemic_uncertainties = []

        while i_sample < len(obs_buf):
            print(i_sample)
            i_to = min(len(obs_buf), i_sample + batch_size)
            obs_act = torch.cat((obs_buf[i_sample:i_to], act_buf[i_sample:i_to]), dim=1)
            next_obs = next_obs_buf[i_sample:i_to]

            (
                prediction,
                means,
                logvars,
                epistemic_uncertainty,
                aleatoric_uncertainty,
            ) = model.get_prediction(obs_act, debug=True)

            model_errors.extend(
                (prediction[:, :-2] - next_obs)
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

                next_obs, _, _, _ = env.step(act.cpu().detach().numpy())

                obs_act = torch.cat((obs.unsqueeze(0), act), dim=1)

                (
                    prediction,
                    means,
                    logvars,
                    epistemic_uncertainty,
                    aleatoric_uncertainty,
                ) = model.get_prediction(obs_act, debug=True)

                model_errors.extend(
                    (prediction[:, :-2].cpu() - next_obs)
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

        # combined_uncertainties = torch.sqrt((raw_aleatoric_uncertainties + 1).square(
        # ) - 1) + torch.sqrt((raw_epistemic_uncertainties + 1).square() - 1)
        combined_uncertainties = aleatoric_uncertainties.clone()
        for i, uncertainty in enumerate(combined_uncertainties):
            if (
                epistemic_uncertainties[i] < aleatoric_uncertainties[i]
                and epistemic_uncertainties[i] < 0.1
            ):
                combined_uncertainties[i] = epistemic_uncertainties[i]

        axes[i_dataset, 0].plot([-1, 2], [-1, 2], color="black")
        axes[i_dataset, 0].scatter(
            model_errors[:n_samples],
            aleatoric_uncertainties[:n_samples],
            s=10,
            alpha=0.3,
            color="blue",
        )
        axes[i_dataset, 0].scatter(
            model_errors[n_samples:],
            aleatoric_uncertainties[n_samples:],
            s=10,
            alpha=0.3,
            color="red",
        )
        if i_dataset == 0:
            axes[i_dataset, 0].set_ylabel("Aleatoric uncertainty", fontsize=12)
            axes[i_dataset, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        else:
            axes[i_dataset, 0].set_yticklabels([])
            for tic in axes[i_dataset, 0].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
                tic.tick2line.set_visible(False)

        for tic in axes[i_dataset, 0].xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
        # axes[i_dataset, 0].set_xticklabels([])
        # axes[i_dataset, 0].set_title(dataset_name)
        axes[i_dataset, 0].set_xlim([0, 1])
        axes[i_dataset, 0].set_ylim([0, 1])

        axes[i_dataset, 1].plot([-1, 2], [-1, 2], color="black")
        axes[i_dataset, 1].scatter(
            model_errors[:n_samples],
            epistemic_uncertainties[:n_samples],
            s=10,
            alpha=0.3,
            color="blue",
        )
        axes[i_dataset, 1].scatter(
            model_errors[n_samples:],
            epistemic_uncertainties[n_samples:],
            s=10,
            alpha=0.3,
            color="red",
        )
        axes[i_dataset, 1].set_xlabel("Model error")
        axes[i_dataset, 0].set_xlabel("Model error")
        if i_dataset == 0:
            axes[i_dataset, 1].set_ylabel("Epistemic uncertainty", fontsize=12)
        else:
            axes[i_dataset, 1].set_yticklabels([])
            for tic in axes[i_dataset, 1].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
                tic.tick2line.set_visible(False)
        axes[i_dataset, 1].set_xlim([0, 1])
        axes[i_dataset, 1].set_ylim([0, 1])

    for ax in axes.flat:
        ax.grid(b=True, alpha=0.5, linestyle="--")

    fig.subplots_adjust(
        top=0.85, bottom=0.085, left=0.07, right=0.985, hspace=0.12, wspace=0.2
    )

    plt.show()
