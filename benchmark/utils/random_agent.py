

class RandomAgent():
    def __init__(self, env):
        self.act_space = env.action_space

    def eval(self):
        pass

    def get_action(self):
        return self.act_space.sample()
