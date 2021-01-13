import torch


class antmaze_selector():

    def __init__(self, buffer):
        goal_state_idxs = buffer.rew_buf > 0
        self.goal_states = buffer.obs_buf[goal_state_idxs][:, :2]

        if len(self.goal_states) == 0:
            raise ValueError("No goal states in buffer.")

        self.max_distance = -1

        for goal_state in self.goal_states:
            max_distance = torch.square(buffer.obs_buf[:, :2] - goal_state)
            max_distance = torch.sum(max_distance, dim=1)
            max_distance = torch.sqrt(max_distance).max()
            if max_distance > self.max_distance:
                self.max_distance = max_distance

        self.progress = 0.1

    def select(self, obs, obs2, act, rew, done):

        max_distance = self.max_distance * self.progress
        goal_state = self.goal_states[torch.randint(len(self.goal_states), (1,))]

        distances = torch.sqrt(torch.sum(torch.square(obs[:, :2] - goal_state), dim=1))

        idxs = distances < max_distance

        return obs[idxs], obs2[idxs], act[idxs], rew[idxs], done[idxs]
