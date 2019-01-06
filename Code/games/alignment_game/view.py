import tkinter as tk
import tkinter.font as tk_font

from . import Tiles, AlignmentGame, TwoPlayerAlignmentGame


class AlignmentGameView(tk.Canvas):

    PLAYER_PADDING = 0.15

    def __init__(self, parent, instance=None):
        tk.Canvas.__init__(self, parent, width=1, height=1)

        self.score_font = tk_font.Font(family="consolas", size="35", weight="normal")  # Cannot declare statically

        self.bind("<Configure>", self.on_resize)
        self.instance = None
        self.geometry = None
        self.tile_width = 0
        self.tile_height = 0
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.set_game_instance(instance)

    def set_game_instance(self, instance):
        self.instance = instance
        self.geometry = {}
        self.delete("all")
        self.geometry["player"] = self.create_oval(0, 0, 0, 0, fill=Colours.PLAYER, outline="")
        self.geometry["score"] = self.create_text(self.width - self.tile_width / 2 + self.score_font.measure("0") / 2,
                                                  self.tile_height / 2, anchor="e", fill=Colours.SCORE,
                                                  font=self.score_font, text="")

        if instance is None:
            self.tile_width = 0
            self.tile_height = 0
        else:
            self.tile_width = float(self.width) / self.instance.view_distance
            self.tile_height = float(self.height) / self.instance.num_lanes
            self.geometry["tiles"] = []
            for x in range(self.instance.view_distance):
                self.geometry["tiles"].append([])
                for y in range(self.instance.num_lanes):
                    self.geometry["tiles"][x].append(
                        self.create_rectangle(x * self.tile_width,
                                              y * self.tile_height,
                                              (x + 1) * self.tile_width,
                                              (y + 1) * self.tile_height, width=0))

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
        self.coords(self.geometry["player"], (0, 0, 0, 0))
        score_x = self.width - self.tile_width / 2 + self.score_font.measure("0") / 2
        self.coords(self.geometry["score"], score_x, self.tile_height / 2)
        self.itemconfig(self.geometry["score"], text=str(self.instance.get_score()))

        state = self.instance.get_state()
        if isinstance(self.instance, AlignmentGame):
            board_1 = state[:self.instance.view_distance]
            board_2 = state[self.instance.view_distance:]
        elif isinstance(self.instance, TwoPlayerAlignmentGame):
            board_1 = state[0]
            board_2 = state[1]
        else:
            raise ValueError("Invalid game instance given to view")
        for x in range(self.instance.view_distance):
            for y in range(self.instance.num_lanes):
                tile_colour = Colours.FLOOR
                if board_1[x][y] == Tiles.WALL:
                    tile_colour = Colours.BOARD_1_WALL
                elif board_2[x][y] == Tiles.WALL:
                    tile_colour = Colours.BOARD_2_WALL
                elif board_1[x][y] == Tiles.PLAYER:
                    self.coords(self.geometry["player"], ((x + self.PLAYER_PADDING) * self.tile_width,
                                                          (y + self.PLAYER_PADDING) * self.tile_height,
                                                          (x + 1 - self.PLAYER_PADDING) * self.tile_width,
                                                          (y + 1 - self.PLAYER_PADDING) * self.tile_height))

                self.itemconfig(self.geometry["tiles"][x][y], fill=tile_colour)
        self.tag_raise(self.geometry["player"])
        self.tag_raise(self.geometry["score"])

    def pack(self):
        super().pack(fill=tk.BOTH, expand=tk.YES)


class Colours:
    FLOOR = "#0f164e"
    BOARD_1_WALL = "#c78787"
    BOARD_2_WALL = "#5b8184"
    PLAYER = "#ffe675"
    SCORE = "#f7f4e3"
