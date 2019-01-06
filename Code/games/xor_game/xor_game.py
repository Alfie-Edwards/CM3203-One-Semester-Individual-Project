from random import Random

from games.game_template import AbstractGameInstance

random = Random()


class XorGame(AbstractGameInstance):

    def __init__(self):
        self.state = None
        self.terminated = None
        self.score = None
        self.reset()

    def do_action(self, action):
        target = 1 if self.state[0] != self.state[1] else 0
        if action == target:
            self.score += 10
        self.terminated = True

    def get_state(self):
        return self.state

    def get_score(self):
        return self.score

    def is_terminated(self):
        return self.terminated

    def reset(self):
        self.state = (random.choice([0, 1]), random.choice([0, 1]))
        self.terminated = False
        self.score = 0
