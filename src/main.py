import tkinter as tk
from video_test import *


window = tk.Tk()

def StartButton():
    Start()

def StopButton():
    Stop()

"""
button = tk.Button(
    text = "Deploy",
    width = 500,
    height = 500,
    command = buttonPressed
)

button.pack()
"""

# create the main sections of the layout, 
# and lay them out
top = tk.Frame(window)
bottom = tk.Frame(window)
top.pack(side=tk.TOP)
bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# create the widgets for the top part of the GUI,
# and lay them out
b = tk.Button(window, text="Start", width=10, height=2, command=StartButton)
c = tk.Button(window, text="Stop", width=10, height=2, command=StopButton)
b.pack(in_=top, side=tk.LEFT)
c.pack(in_=top, side=tk.LEFT)

window.mainloop()