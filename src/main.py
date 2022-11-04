import tkinter as tk
from video import *

window = tk.Tk()
window.geometry("500x500");

def buttonPressed():
    Start()

button = tk.Button
(
    text="Deploy",
    width=500,
    height=500,
    command=buttonPressed
);

button.pack()

window.mainloop()
