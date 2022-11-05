from tkinter import *
from video_test import *
import yaml
import customtkinter as ctk

battery = getBattery()

ctk.set_appearance_mode("dark")
window = ctk.CTk()
window.geometry("400x150")
window.resizable(False, False)

def StartButton():

    name = e1.get()
    email = e2.get()

    if not name or not email:
        return

    dict = {
        "name": name,
        "email": email
    }

    with open("sender.yml", 'w') as file:
        yaml.dump(dict, file)

    Start()

def StopButton():
    Stop()

b = ctk.CTkButton(window, text="Start", width=120, height=40, command=StartButton).grid(row=0, column=0)
c = ctk.CTkButton(window, text="Stop", width=120, height=40, command=StopButton).grid(row=0, column=1)

ctk.CTkLabel(window, text="Tvoje jméno").grid(row=1)
ctk.CTkLabel(window, text="email").grid(row=2)
ctk.CTkLabel(window, text=str(battery) + " %").grid(row=3)

e1 = ctk.CTkEntry(window)
e2 = ctk.CTkEntry(window)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

if battery <= 20:
    messagebox.showwarning("warning","Warning")

window.mainloop()