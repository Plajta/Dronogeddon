from tkinter import *
#from video_test import *
import yaml
import customtkinter as ctk

ctk.set_appearance_mode("dark")
window = ctk.CTk()
window.geometry("400x100")

def StartButton():

    name = e1.get()
    email = e2.get()

    dict = {
        "name": name,
        "email": email
    }

    with open("sender.yml", 'w') as file:
        yaml.dump(dict, file)

    #Start()

def StopButton():
    #Stop()
    pass

b = ctk.CTkButton(window, text="Start", width=120, height=40, command=StartButton).grid(row=0, column=0)
c = ctk.CTkButton(window, text="Stop", width=120, height=40, command=StopButton).grid(row=0, column=1)

ctk.CTkLabel(window, text="Tvoje jméno").grid(row=1)
ctk.CTkLabel(window, text="email").grid(row=2)

e1 = ctk.CTkEntry(window)
e2 = ctk.CTkEntry(window)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

window.mainloop()