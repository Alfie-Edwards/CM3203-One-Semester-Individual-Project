import copy
from . import Tiles, Directions


class PathGameModel:

    def __init__(self, board, player_x, player_y, goal_x, goal_y):
        self.board = board
        self.board_width = len(board)
        self.board_height = len(board[0])
        self.player_x = player_x
        self.player_y = player_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.dead = False
        self.goal_reached = False
        self.score = 0

        for i in range(self.board_width):
            for j in range(self.board_height):
                if self.board[i][j] != Tiles.FLOOR and self.board[i][j] != Tiles.WALL:
                    raise Exception("INVALID GAME STATE: Board contains tiles with unknown values")

        if (self.player_x < 0 or self.player_x >= self.board_width or
                self.player_y < 0 or self.player_y >= self.board_height):
            raise Exception("INVALID GAME STATE: Player start position is outside of the board")

        if (self.goal_x < 0 or self.goal_x >= self.board_width or
                self.goal_y < 0 or self.goal_y >= self.board_height):
            raise Exception("INVALID GAME STATE: Goal is outside of the board")

        if self.board[self.player_x][self.player_y] != Tiles.FLOOR:
            raise Exception("INVALID GAME STATE: Player must start the game on a floor tile")

        if self.board[self.goal_x][self.goal_y] != Tiles.FLOOR:
            raise Exception("INVALID GAME STATE: Goal must be on a floor tile")

    def move(self, direction):
        if self.dead or self.goal_reached:
            return 0

        self.score -= 1
        new_player_x = self.player_x
        new_player_y = self.player_y

        if direction == Directions.RIGHT:
            new_player_x += 1
        elif direction == Directions.UP:
            new_player_y -= 1
        elif direction == Directions.LEFT:
            new_player_x -= 1
        elif direction == Directions.DOWN:
            new_player_y += 1
        else:
            raise Exception("ERROR: Invalid direction specified: ", direction)

        if (new_player_x < 0 or new_player_x >= self.board_width or
                new_player_y < 0 or new_player_y >= self.board_height or
                self.board[new_player_x][new_player_y] == Tiles.WALL):
            self.score -= 5
            return

        if new_player_x == self.goal_x and new_player_y == self.goal_y:
            self.goal_reached = True
            self.score += 100

        self.player_x = new_player_x
        self.player_y = new_player_y

    def clone(self):
        return PathGameModel(copy.deepcopy(self.board), self.player_x, self.player_y, self.goal_x, self.goal_y)
