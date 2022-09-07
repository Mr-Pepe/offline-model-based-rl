from typing import Optional

import gym
import torch


class RandomAgent:
    def __init__(self, env: gym.Env, device: str = "cpu") -> None:
        self.training = False
        self.act_space = env.action_space
        self.device = device

    def eval(self) -> None:
        """Dummy function."""
        pass

    def act(self, unused_observation: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.as_tensor(
            self.act_space.sample().reshape((1, -1)), device=torch.device(self.device)
        )
