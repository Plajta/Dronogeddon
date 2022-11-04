import tkinter as tk
from video import *

window = tk.Tk()
window.geometry("500x500");

def buttonPressed():
    Start()

<<<<<<< HEAD
button = tk.Button(
    text="Deploy",
    width=500,
    height=500,
    command=buttonPressed
)
=======
button = tk.Button
(
    text = "Deploy",
    width = 500,
    height = 500,
    command = buttonPressed
);
>>>>>>> parent of c02eec5 (removed spaces causing errors)

button.pack()

window.mainloop()
