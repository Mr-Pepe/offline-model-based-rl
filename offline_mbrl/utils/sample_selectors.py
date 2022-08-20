import torch
from offline_mbrl.utils.envs import ANTMAZE_MEDIUM_ENVS, ANTMAZE_UMAZE_ENVS


class antmaze_selector:
    def __init__(self, buffer):
        self.goal_state_idxs = torch.nonzero(buffer.rew_buf > 0, as_tuple=False)

        if self.goal_state_idxs.numel() == 0:
            raise ValueError("No goal states in buffer.")

        print("Found {} goal states.".format(self.goal_state_idxs.numel()))

        mean_goal = buffer.obs_buf[self.goal_state_idxs, :2].mean(dim=0)

        self.max_distance = torch.sqrt(
            torch.sum(torch.square(buffer.obs_buf[:, :2] - mean_goal), dim=1)
        ).max()

        print("Maximum distance in curriculum: {}".format(self.max_distance))

        self.progress = 0.1

    def select(self, buffer):

        max_distance = self.max_distance * self.progress
        goal_state = buffer.obs_buf[
            self.goal_state_idxs[torch.randint(self.goal_state_idxs.numel(), (1,))]
        ].view(-1)[:2]

        distances = torch.sqrt(
            torch.sum(torch.square(buffer.obs_buf[:, :2] - goal_state), dim=1)
        )

        idxs = distances < max_distance

        idxs = torch.logical_and(idxs, buffer.done_buf == 0)

        return torch.nonzero(idxs, as_tuple=False)


def get_selector(env_name):
    if env_name in ANTMAZE_MEDIUM_ENVS or env_name in ANTMAZE_UMAZE_ENVS:
        return antmaze_selector
    else:
        return None
