import tkinter as tk

from games.alignment_game import AlignmentGame, AlignmentGameView

num_lanes = 5
wall_spacing = 6
view_distance = 5


window = tk.Tk()
window.title("Path pathgame")
window.geometry("640x640")
instance = AlignmentGame(num_lanes, view_distance, wall_spacing)
view = AlignmentGameView(window, instance)
view.pack()

window.bind("<Up>", lambda x: (instance.do_action(0), view.update()))
window.bind("<Right>", lambda x: (instance.do_action(2), view.update()))
window.bind("<Down>", lambda x: (instance.do_action(1), view.update()))

window.mainloop()