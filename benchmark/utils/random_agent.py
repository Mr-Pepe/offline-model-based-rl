

class RandomAgent():
    def __init__(self, env):
        self.training = False
        self.act_space = env.action_space

    def eval(self):
        pass

    def get_action(self, obs=0):
        return self.act_space.sample().reshape((1, -1))
