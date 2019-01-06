import tkinter as tk

from games.path_game import random_generation, PathGameModel, PathGameView, Directions

board_height = 7
board_width = 7
goal_x = board_width - 1
goal_y = board_height - 1
start_x = 0
start_y = 0
x = start_x
y = start_y

board = random_generation.single_path_board(board_width, board_height, start_x, start_y, goal_x, goal_y)


window = tk.Tk()
window.title("Path pathgame")
window.geometry("640x640")
model = PathGameModel(board, start_x, start_y, goal_x, goal_y)
view = PathGameView(window, model)
view.pack()

window.bind("<Right>", lambda x: (model.move(Directions.RIGHT), view.update()))
window.bind("<Up>", lambda x: (model.move(Directions.UP), view.update()))
window.bind("<Left>", lambda x: (model.move(Directions.LEFT), view.update()))
window.bind("<Down>", lambda x: (model.move(Directions.DOWN), view.update()))

window.mainloop()