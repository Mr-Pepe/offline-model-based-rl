import torch


class RandomAgent:
    def __init__(self, env, device="cpu"):
        self.training = False
        self.act_space = env.action_space
        self.device = device

    def eval(self):
        pass

    def act(self, obs=0):
        return torch.as_tensor(
            self.act_space.sample().reshape((1, -1)), device=self.device
        )
