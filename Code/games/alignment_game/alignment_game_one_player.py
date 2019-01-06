from games.game_template import AbstractGameInstance
from random import Random

from games.alignment_game.constants import Tiles

random = Random()


class AlignmentGame(AbstractGameInstance):
    def __init__(self, num_lanes, view_distance, wall_spacing):
        self.num_lanes = num_lanes
        self.view_distance = view_distance
        self.wall_spacing = wall_spacing
        self.blank_column = tuple(0 for _ in range(self.num_lanes))

        self.dead = None
        self.current_turn = None
        self.player_lane = None
        self.board_1 = None
        self.board_2 = None
        self.reset()

    def get_state(self):
        return tuple(self.board_1 + self.board_2)

    def do_action(self, action):
        if self.dead:
            return

        self.__advance_board()
        if action == 0:
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 1:
            self.player_lane = min(self.num_lanes - 1, self.player_lane + 1)

        if (self.board_1[0][self.player_lane] == Tiles.WALL or
                self.board_2[0][self.player_lane] == Tiles.WALL):
            self.dead = True
        else:
            self.board_1[0] = self.board_1[0][:self.player_lane] + (Tiles.PLAYER,) + self.board_1[0][self.player_lane + 1:]
            self.board_2[0] = self.board_2[0][:self.player_lane] + (Tiles.PLAYER,) + self.board_2[0][self.player_lane + 1:]

        self.current_turn += 1

    def __advance_board(self):
        if self.current_turn % (self.wall_spacing + 1) == 0:
            hole_position = random.choice(range(self.num_lanes))
            col_1 = []
            col_2 = []
            for i in range(self.num_lanes):
                if i == hole_position:
                    col_1.append(Tiles.FLOOR)
                    col_2.append(Tiles.FLOOR)
                elif random.random() < 0.5:
                    col_1.append(Tiles.WALL)
                    col_2.append(Tiles.FLOOR)
                else:
                    col_1.append(Tiles.FLOOR)
                    col_2.append(Tiles.WALL)
            self.board_1.append(tuple(col_1))
            self.board_2.append(tuple(col_2))
        else:
            self.board_1.append(self.blank_column)
            self.board_2.append(self.blank_column)
        self.board_1.pop(0)
        self.board_2.pop(0)

    def get_score(self):
        # One point for each wall passed
        return max(0, self.current_turn - self.view_distance + self.wall_spacing) // (self.wall_spacing + 1)

    def is_terminated(self):
        return self.dead

    def reset(self):
        self.dead = False
        self.current_turn = 0
        self.player_lane = round(self.num_lanes / 2)
        self.board_1 = [self.blank_column for _ in range(self.view_distance)]
        self.board_2 = [self.blank_column for _ in range(self.view_distance)]
        self.board_1[0] = self.board_1[0][:self.player_lane] + (Tiles.PLAYER,) + self.board_1[0][self.player_lane + 1:]
        self.board_2[0] = self.board_2[0][:self.player_lane] + (Tiles.PLAYER,) + self.board_2[0][self.player_lane + 1:]