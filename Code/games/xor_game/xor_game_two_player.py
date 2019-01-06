from random import Random

from games.game_template import AbstractGameInstance

random = Random()


class TwoPlayerXorGame(AbstractGameInstance):

    def __init__(self):
        self.state = None
        self.terminated = None
        self.score = None
        self.reset()

    def do_action(self, action):
        action_1 = action[0]
        action_2 = action[1]

        target_1 = 1 if self.state[0][0] != self.state[1][1] else 0
        target_2 = 1 if self.state[0][1] != self.state[1][0] else 0

        if action_1 == target_1 and action_2 == target_2:
            self.score = 10
        else:
            self.score = 0
        self.terminated = True

    def get_state(self):
        return self.state

    def get_score(self):
        return self.score

    def is_terminated(self):
        return self.terminated

    def reset(self):
        self.state = ((random.choice([0, 1]), random.choice([0, 1])), (random.choice([0, 1]), random.choice([0, 1])))
        self.terminated = False
        self.score = 0
