from games.path_game.constants import Tiles, Directions
from random import Random

random = Random()


def single_path_board(board_width, board_height, start_x, start_y, goal_x, goal_y):
    direction_pool = []
    while True:
        if len(direction_pool) == 0:
            board = [[Tiles.WALL for y in range(board_height)] for x in range(board_width)]
            x = start_x
            y = start_y
            board[start_x][start_y] = Tiles.FLOOR
            direction_pool = [Directions.RIGHT, Directions.UP, Directions.LEFT, Directions.DOWN]

        if x == goal_x and y == goal_y:
            return board

        direction = random.choice(direction_pool)
        direction_pool.remove(direction)

        if direction == Directions.RIGHT and x + 2 < board_width and board[x + 2][y] == Tiles.WALL:
                board[x + 1][y] = Tiles.FLOOR
                board[x + 2][y] = Tiles.FLOOR
                x += 2
        elif direction == Directions.UP and y - 2 >= 0 and board[x][y - 2] == Tiles.WALL:
                board[x][y - 1] = Tiles.FLOOR
                board[x][y - 2] = Tiles.FLOOR
                y -= 2
        elif direction == Directions.LEFT and x - 2 >= 0 and board[x - 2][y] == Tiles.WALL:
                board[x - 1][y] = Tiles.FLOOR
                board[x - 2][y] = Tiles.FLOOR
                x -= 2
        elif direction == Directions.DOWN and y + 2 < board_height and board[x][y + 2] == Tiles.WALL:
                board[x][y + 1] = Tiles.FLOOR
                board[x][y + 2] = Tiles.FLOOR
                y += 2
        else:
            continue

        direction_pool = [Directions.RIGHT, Directions.UP, Directions.LEFT, Directions.DOWN]


def depth_first_tree_maze(board_width, board_height, goal_x, goal_y):
    board = [[Tiles.WALL for y in range(board_height)] for x in range(board_width)]
    x = goal_x
    y = goal_y

    board[goal_x][goal_y] = Tiles.FLOOR
    position_stack = [(x, y)]
    direction_pool = [Directions.RIGHT, Directions.UP, Directions.LEFT, Directions.DOWN]

    while True:
        if not direction_pool:
            if x == goal_x and y == goal_y:
                return board
            x, y = position_stack.pop()
            direction_pool = [Directions.RIGHT, Directions.UP, Directions.LEFT, Directions.DOWN]

        direction = random.choice(direction_pool)
        direction_pool.remove(direction)

        if direction == Directions.RIGHT and x + 2 < board_width and board[x + 2][y] == Tiles.WALL:
            board[x + 1][y] = Tiles.FLOOR
            board[x + 2][y] = Tiles.FLOOR
            x += 2
        elif direction == Directions.UP and y - 2 >= 0 and board[x][y - 2] == Tiles.WALL:
            board[x][y - 1] = Tiles.FLOOR
            board[x][y - 2] = Tiles.FLOOR
            y -= 2
        elif direction == Directions.LEFT and x - 2 >= 0 and board[x - 2][y] == Tiles.WALL:
            board[x - 1][y] = Tiles.FLOOR
            board[x - 2][y] = Tiles.FLOOR
            x -= 2
        elif direction == Directions.DOWN and y + 2 < board_height and board[x][y + 2] == Tiles.WALL:
            board[x][y + 1] = Tiles.FLOOR
            board[x][y + 2] = Tiles.FLOOR
            y += 2
        else:
            continue

        position_stack.append((x, y))
        direction_pool = [Directions.RIGHT, Directions.UP, Directions.LEFT, Directions.DOWN]


def random_start_position(board, goal_x, goal_y, only_even=True):
    width = len(board)
    height = len(board[0])
    x = goal_x
    y = goal_y
    while board[x][y] == Tiles.WALL or x == goal_x or y == goal_y:
        if only_even:
            x = int(random.random() * width / 2) * 2
            y = int(random.random() * height / 2) * 2
        else:
            x = int(random.random() * width)
            y = int(random.random() * height)

    return x, y