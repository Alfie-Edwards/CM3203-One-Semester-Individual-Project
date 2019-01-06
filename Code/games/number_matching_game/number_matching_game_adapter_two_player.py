from games.game_template import AbstractGameInstance


class TwoPlayerNumberMatchingGameAdapter(AbstractGameInstance):
    
    def __init__(self, number_matching_game_model, turn_limit=None):
        self.number_matching_game_model = number_matching_game_model
        self.turn_limit = turn_limit
        self.turns_remaining = None
        self.selection_1 = None
        self.selection_2 = None
        self.score = None
        self.reset()

    def do_action(self, action):
        action_1, action_2 = action
        self.selection_1[action_1] = not self.selection_1[action_1]
        self.selection_2[action_2] = not self.selection_2[action_2]
        self.score = self.number_matching_game_model.get_score(self.selection_1, self.selection_2)

        if self.turn_limit is not None:
            self.turns_remaining -= 1

    def get_state(self):
        options_1 = tuple(self.number_matching_game_model.options_1)
        options_2 = tuple(self.number_matching_game_model.options_2)
        selection_1 = tuple(int(value) for value in self.selection_1)
        selection_2 = tuple(int(value) for value in self.selection_2)
        return ((options_1, selection_1), (options_2, selection_2))

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
