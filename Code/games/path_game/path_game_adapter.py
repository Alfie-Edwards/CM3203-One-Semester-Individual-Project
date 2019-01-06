from games.game_template import AbstractGameInstance
from . import Tiles


class PathGameAdapter(AbstractGameInstance):

    def __init__(self, path_game_model, use_double_moves=True):
        self.base_path_game_model = path_game_model
        self.path_game_model = self.base_path_game_model.clone()
        self.use_double_moves = use_double_moves

    def do_action(self, action):
        self.path_game_model.move(action)
        if self.use_double_moves:
            self.path_game_model.move(action)

    def get_state(self):
        board = self.path_game_model.board
        remapped_board = []
        for x in range(self.path_game_model.board_width):
            if x == self.path_game_model.player_x:
                column = list(board[x])
                column[self.path_game_model.player_y] = Tiles.PLAYER
            else:
                column = board[x]
            remapped_board.append(tuple(column))
        remapped_board = tuple(remapped_board)
        return remapped_board

    def get_score(self):
        return self.path_game_model.score

    def is_terminated(self):
        return self.path_game_model.dead or self.path_game_model.goal_reached

    def reset(self):
        self.path_game_model = self.base_path_game_model.clone()

