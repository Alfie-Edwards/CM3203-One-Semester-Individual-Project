from games.game_template import AbstractGameInstance
from . import Tiles


class TwoPlayerPathGameAdapter(AbstractGameInstance):

    WALL_TILE_VALUE = -1
    FLOOR_TILE_VALUE = 0
    PLAYER_TILE_VALUE = 1

    def __init__(self, path_game_model_1, path_game_model_2, use_double_moves=True):
        self.base_path_game_model_1 = path_game_model_1
        self.base_path_game_model_2 = path_game_model_2
        self.path_game_model_1 = self.base_path_game_model_1.clone()
        self.path_game_model_2 = self.base_path_game_model_2.clone()
        self.use_double_moves = use_double_moves

    def do_action(self, action):
        action_1, action_2 = action
        self.path_game_model_1.move(action_2)
        self.path_game_model_2.move(action_1)
        if self.use_double_moves:
            self.path_game_model_1.move(action_2)
            self.path_game_model_2.move(action_1)

    def get_state(self):
        state_1 = self.process_state(self.path_game_model_1.board,
                                     self.path_game_model_1.player_x,
                                     self.path_game_model_1.player_y)
        state_2 = self.process_state(self.path_game_model_2.board,
                                     self.path_game_model_2.player_x,
                                     self.path_game_model_2.player_y)
        return state_1, state_2

    def get_score(self):
        return self.path_game_model_1.score + self.path_game_model_2.score

    def is_terminated(self):
        return ((self.path_game_model_1.dead or self.path_game_model_1.goal_reached) and
                (self.path_game_model_2.dead or self.path_game_model_2.goal_reached))

    def reset(self):
        self.path_game_model_1 = self.base_path_game_model_1.clone()
        self.path_game_model_2 = self.base_path_game_model_2.clone()

    @staticmethod
    def map_tile_values(tile):
        if tile == Tiles.WALL:
            return TwoPlayerPathGameAdapter.WALL_TILE_VALUE
        if tile == Tiles.FLOOR:
            return TwoPlayerPathGameAdapter.FLOOR_TILE_VALUE
        raise ValueError("Unknown tile value: " + str(tile))

    @staticmethod
    def process_state(board, player_x, player_y):
        state = []
        for x in range(len(board)):
            column = []
            for y in range(len(board[0])):
                if x == player_x and y == player_y:
                    column.append(TwoPlayerPathGameAdapter.PLAYER_TILE_VALUE)
                else:
                    column.append(TwoPlayerPathGameAdapter.map_tile_values(board[x][y]))
            state.append(tuple(column))
        return tuple(state)
