from games.game_template import AbstractGameInstance


class NumberMatchingGameAdapter(AbstractGameInstance):

    def __init__(self, number_matching_game_model, turn_limit=None):
        self.number_matching_game_model = number_matching_game_model
        self.turn_limit = turn_limit
        self.turns_remaining = None
        self.selection_1 = None
        self.selection_2 = None
        self.score = None
        self.reset()

    def do_action(self, action):
        if action // self.number_matching_game_model.option_count == 0:
            self.selection_1[action] = not self.selection_1[action]
        else:
            option = action - self.number_matching_game_model.option_count
            self.selection_2[option] = not self.selection_2[option]

        self.score = self.number_matching_game_model.get_score(self.selection_1, self.selection_2)

        if self.turn_limit is not None:
            self.turns_remaining -= 1

    def get_state(self):
        options = tuple(self.number_matching_game_model.options_1 + self.number_matching_game_model.options_2)
        selection = tuple(int(value) for value in self.selection_1 + self.selection_2)
        return (options, selection)

    def get_score(self):
        return self.score

    def is_terminated(self):
        if self.turn_limit is not None:
            return self.turns_remaining == 0
        else:
            return self.score == 100

    def reset(self):
        self.number_matching_game_model.new_random_options()
        self.selection_1 = [False] * self.number_matching_game_model.option_count
        self.selection_2 = [False] * self.number_matching_game_model.option_count
        self.turns_remaining = self.turn_limit
        self.score = 0
