import os

import d4rl  # pylint: disable=unused-import
import gym
import torch

from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.modes import (
    ALEATORIC_MODES,
    ALEATORIC_PARTITIONING,
    EPISTEMIC_MODES,
    EXPLICIT_MODES,
)


def get_uncertainty_distribution(env_name, mode, all_stats=False):
    env = gym.make(env_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(
        os.path.join(MODELS_DIR, env_name + "-model.pt"), map_location=device
    )

    buffer, obs_dim, act_dim = load_dataset_from_env(
        env, buffer_device=next(model.parameters()).device
    )

    batch_size = 4096

    explicit_uncertainties = None
    epistemic_uncertainties = None
    aleatoric_uncertainties = None

    for i in range(0, buffer.size, batch_size):
        obs = buffer.obs_buf[i : i + batch_size]
        act = buffer.act_buf[i : i + batch_size]
        obs_act = torch.cat((obs, act), dim=1)

        (
            predictions,
            this_means,
            this_logvars,
            this_explicit_uncertainty,
            this_epistemic_uncertainty,
            this_aleatoric_uncertainty,
            _,
        ) = model.get_prediction(obs_act, mode=ALEATORIC_PARTITIONING, debug=True)

        if explicit_uncertainties is None:
            explicit_uncertainties = this_explicit_uncertainty
            epistemic_uncertainties = this_epistemic_uncertainty
            aleatoric_uncertainties = this_aleatoric_uncertainty
        else:
            explicit_uncertainties = torch.cat(
                (explicit_uncertainties, this_explicit_uncertainty), dim=1
            )
            epistemic_uncertainties = torch.cat(
                (epistemic_uncertainties, this_epistemic_uncertainty)
            )
            aleatoric_uncertainties = torch.cat(
                (aleatoric_uncertainties, this_aleatoric_uncertainty)
            )

    explicit_uncertainties = explicit_uncertainties.mean(dim=0)

    rew_span = max(buffer.rew_buf.max().abs().item(), buffer.rew_buf.min().abs().item())

    print("\nEnvironment: {}\n".format(env_name))
    print("Reward span: {}".format(rew_span))
    print(
        "Aleatoric max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(
            aleatoric_uncertainties.max(),
            aleatoric_uncertainties.mean(),
            aleatoric_uncertainties.std(),
        )
    )
    print(
        "Epistemic max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(
            epistemic_uncertainties.max(),
            epistemic_uncertainties.mean(),
            epistemic_uncertainties.std(),
        )
    )
    print(
        "Explicit  max, mean, std: ({:.6f}, {:.6f}, {:.6f})".format(
            explicit_uncertainties.max(),
            explicit_uncertainties.mean(),
            explicit_uncertainties.std(),
        )
    )

    if all_stats:
        return dict(
            aleatoric=(
                rew_span,
                aleatoric_uncertainties.max().item(),
                aleatoric_uncertainties.mean().item(),
                aleatoric_uncertainties.std().item(),
            ),
            epistemic=(
                rew_span,
                epistemic_uncertainties.max().item(),
                epistemic_uncertainties.mean().item(),
                epistemic_uncertainties.std().item(),
            ),
            explicit=(
                rew_span,
                explicit_uncertainties.max().item(),
                explicit_uncertainties.mean().item(),
                explicit_uncertainties.std().item(),
            ),
        )
    elif mode in ALEATORIC_MODES:
        return (
            rew_span,
            aleatoric_uncertainties.max().item(),
            aleatoric_uncertainties.mean().item(),
            aleatoric_uncertainties.std().item(),
        )
    elif mode in EPISTEMIC_MODES:
        return (
            rew_span,
            epistemic_uncertainties.max().item(),
            epistemic_uncertainties.mean().item(),
            epistemic_uncertainties.std().item(),
        )
    elif mode in EXPLICIT_MODES:
        return (
            rew_span,
            explicit_uncertainties.max().item(),
            explicit_uncertainties.mean().item(),
            explicit_uncertainties.std().item(),
        )


def format_numbers(numbers):
    out = []
    for number in numbers:
        if number < 0.01:
            out.append("{:.2e}".format(number))
        else:
            out.append("{:.2f}".format(number))

    return out


if __name__ == "__main__":
    prefix = "halfcheetah-"
    version = "-v2"

    dataset_names = [
        "random",
        # 'medium-replay',
        # 'medium',
        # 'medium-expert',
    ]

    avg_rew = []
    per_trajectory_rews = []

    latex = ""

    for dataset_name in dataset_names:
        env_name = prefix + dataset_name + version
        print(env_name)
        stats = get_uncertainty_distribution(env_name, "", all_stats=True)

        latex += "& {} & {} & {} & {} & {} & {} & {} \\\\ ".format(
            dataset_name,
            *format_numbers(
                (
                    stats["aleatoric"][2],
                    stats["aleatoric"][3],
                    stats["aleatoric"][1],
                    stats["epistemic"][2],
                    stats["epistemic"][3],
                    stats["epistemic"][1],
                )
            )
        ).replace("e+00", "")

        print(latex)
