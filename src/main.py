import tkinter as tk
from video import *


window = tk.Tk()
window.geometry("500x500");

def StartButton():
    Start()

def StopButton():
    exit(1)

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
b = tk.Button(window, text="Enter", width=10, height=2, command=StartButton)
c = tk.Button(window, text="Clear", width=10, height=2, command=StopButton)
b.pack(in_=top, side=tk.LEFT)
c.pack(in_=top, side=tk.LEFT)

window.mainloop()