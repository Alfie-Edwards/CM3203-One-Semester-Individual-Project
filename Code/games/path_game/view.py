import tkinter as tk
from . import Tiles


class PathGameView(tk.Canvas):

    border_thickness = 0.05
    player_padding = 0.15
    goal_padding = 0.25

    def __init__(self, parent, model):
        tk.Canvas.__init__(self, parent, width=1, height=1)
        self.model = model
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

        self.tile_width = float(self.width) / self.model.board_width
        self.tile_height = float(self.height) / self.model.board_height
        self.geometry = {}
        self.initialise_geometry()

    def initialise_geometry(self):
        self.geometry["tiles"] = []
        for x in range(self.model.board_width):
            self.geometry["tiles"].append([])
            for y in range(self.model.board_height):
                self.geometry["tiles"][x].append(
                    (self.create_rectangle(x * self.tile_width,
                                           y * self.tile_height,
                                           (x + 1) * self.tile_width,
                                           (y + 1) * self.tile_height, width=0),
                     self.create_rectangle((x + self.border_thickness) * self.tile_width,
                                           (y + self.border_thickness) * self.tile_height,
                                           (x - self.border_thickness + 1) * self.tile_width,
                                           (y - self.border_thickness + 1) * self.tile_height, width=0)))

        self.geometry["player"] = self.create_oval(0, 0, 0, 0, fill=Colours.PLAYER, outline="")
        self.geometry["goal"] = (self.create_line(0, 0, 0, 0, fill=Colours.GOAL, width=8),
                                 self.create_line(0, 0, 0, 0, fill=Colours.GOAL, width=8))
        self.update()

    def on_resize(self, event):
        x_scale = float(event.width) / self.width
        y_scale = float(event.height) / self.height
        self.tile_width *= x_scale
        self.tile_height *= y_scale
        self.width = event.width
        self.height = event.height
        self.scale("all", 0, 0, x_scale, y_scale)

    def update(self):
        for x in range(self.model.board_width):
            for y in range(self.model.board_height):
                tile_colour = Colours.FLOOR if self.model.board[x][y] == Tiles.FLOOR else Colours.WALL
                border_colour = Colours.FLOOR_BORDER if self.model.board[x][y] == Tiles.FLOOR else Colours.WALL_BORDER
                self.itemconfig(self.geometry["tiles"][x][y][0], fill=border_colour)
                self.itemconfig(self.geometry["tiles"][x][y][1], fill=tile_colour)

        self.coords(self.geometry["player"], ((self.model.player_x + self.player_padding) * self.tile_width,
                                              (self.model.player_y + self.player_padding) * self.tile_height,
                                              (self.model.player_x + 1 - self.player_padding) * self.tile_width,
                                              (self.model.player_y + 1 - self.player_padding) * self.tile_height))
        self.coords(self.geometry["goal"][0], ((self.model.goal_x + self.goal_padding) * self.tile_width,
                                               (self.model.goal_y + self.goal_padding) * self.tile_height,
                                               (self.model.goal_x + 1 - self.goal_padding) * self.tile_width,
                                               (self.model.goal_y + 1 - self.goal_padding) * self.tile_height))
        self.coords(self.geometry["goal"][1], ((self.model.goal_x + self.goal_padding) * self.tile_width,
                                               (self.model.goal_y + 1 - self.goal_padding) * self.tile_height,
                                               (self.model.goal_x + 1 - self.goal_padding) * self.tile_width,
                                               (self.model.goal_y + self.goal_padding) * self.tile_height))

    def pack(self):
        super().pack(fill=tk.BOTH, expand=tk.YES)


class Colours:
    FLOOR = "#a8d360"
    WALL = "#b3e7e6"
    PLAYER = "#f7cf84"
    GOAL = "#d36747"
    FLOOR_BORDER = "#c1e69c"
    WALL_BORDER = "#c7eeec"
