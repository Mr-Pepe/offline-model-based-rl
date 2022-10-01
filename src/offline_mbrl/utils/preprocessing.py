from functools import partial
from typing import Callable, Optional

# pylint: disable=unused-import
import gym
import torch
from d4rl.offline_env import OfflineEnv

from offline_mbrl.utils.envs import ALL_ENVS


def get_preprocessing_function(
    env_name: str, device: str = "cpu"
) -> Optional[Callable]:
    """Retrieves a preprocessing function for a given environment.

    The preprocessing function performs normalization based on the mean and standard
    deviation of the offline dataset for an environment.

    The input does not get normalized if no preprocessing function could be found.

    Args:
        env_name (str): The environment name.
        device (str, optional): The PyTorch device. Defaults to "cpu".

    Returns:
        Optional[Callable]: The preprocessing function for that environment.
    """
    if env_name not in ALL_ENVS:
        return partial(_preprocess, None, None)

    env: OfflineEnv = gym.make(env_name)
    dataset = env.get_dataset()

    obs_act = torch.cat(
        (torch.as_tensor(dataset["observations"]), torch.as_tensor(dataset["actions"])),
        dim=1,
    )

    mean = obs_act.mean(dim=0).to(device)
    std = obs_act.std(dim=0).to(device)

    return partial(_preprocess, mean, std)


def _preprocess(
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    obs_act: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    """Preprocess a batch of concatenated observations and actions.

    Uses normalization for preprocessing with statistics defined in :code:`mean` and
    :code:`std`.

    Args:
        mean (torch.Tensor): The mean value to use for normalization.
        std (torch.Tensor): The std value to use for normalization.
        obs_act (torch.Tensor): A tensor containing samples of concatenated observations
            and actions.
        detach (bool, optional): Whether to detach the samples before preprocessing.
            Defaults to True.

    Returns:
        torch.Tensor: A tensor of the same shape as obs_act with normalized samples.
    """
    if detach:
        obs_act = obs_act.detach().clone()

    if mean is not None and std is not None:
        # This allows to preprocess an observation without action
        length = obs_act.shape[-1]

        obs_act -= mean[:length]
        obs_act /= std[:length]

    return obs_act
