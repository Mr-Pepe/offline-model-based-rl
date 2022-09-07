#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains a function to get the uncertainty distribution of a model on a dataset."""


# pylint: disable=consider-using-f-string


import os

import d4rl  # pylint: disable=unused-import
import torch

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.modes import ALEATORIC_MODES, EPISTEMIC_MODES


def get_uncertainty_distribution(
    env_name: str, mode: str
) -> tuple[float, float, float, float]:
    """Calculates the uncertainty distribution of an environment model on a dataset.

    Loads the specific dataset and the corresponding environment model. The model must
    have been trained and saved to the :py:const:`.MODELS_DIR`.

    Args:
        env_name (str): The environment name.
        mode (str): The mode to compute the uncertainty distribution for.

    Returns:
        tuple[float, float, float, float]: The reward span and the max, mean, and
            standard deviation value that the model predicted for the dataset.
    """
    if mode not in (*ALEATORIC_MODES, *EPISTEMIC_MODES):
        raise ValueError(
            f"Mode must be in {ALEATORIC_MODES} or {EPISTEMIC_MODES} but got '{mode}'."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model: EnvironmentModel = torch.load(
        os.path.join(MODELS_DIR, env_name + "-model.pt"), map_location=device
    )

    buffer, _, _ = load_dataset_from_env(env_name, buffer_device=device)

    batch_size = 4096

    epistemic_uncertainties = None
    aleatoric_uncertainties = None

    for batch_start_index in range(0, buffer.size, batch_size):
        obs = buffer.obs_buf[batch_start_index : batch_start_index + batch_size]
        act = buffer.act_buf[batch_start_index : batch_start_index + batch_size]
        obs_act = torch.cat((obs, act), dim=1)

        (
            _,
            _,
            _,
            this_epistemic_uncertainty,
            this_aleatoric_uncertainty,
        ) = model.get_prediction(obs_act, debug=True)

        if epistemic_uncertainties is None:
            epistemic_uncertainties = this_epistemic_uncertainty
            aleatoric_uncertainties = this_aleatoric_uncertainty
        else:
            epistemic_uncertainties = torch.cat(
                (epistemic_uncertainties, this_epistemic_uncertainty)
            )
            aleatoric_uncertainties = torch.cat(
                (aleatoric_uncertainties, this_aleatoric_uncertainty)
            )

    rew_span = max(buffer.rew_buf.max().abs().item(), buffer.rew_buf.min().abs().item())

    assert aleatoric_uncertainties is not None
    assert epistemic_uncertainties is not None

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

    if mode in ALEATORIC_MODES:
        return (
            rew_span,
            aleatoric_uncertainties.max().item(),
            aleatoric_uncertainties.mean().item(),
            aleatoric_uncertainties.std().item(),
        )
    return (
        rew_span,
        epistemic_uncertainties.max().item(),
        epistemic_uncertainties.mean().item(),
        epistemic_uncertainties.std().item(),
    )
