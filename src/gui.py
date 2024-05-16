import tkinter as tk
from PIL import Image, ImageTk
import cv2
import multiprocessing
import time

class Window:
    def __init__(self, win_name, dim=[700, 700]):
        self.win = tk.Tk()
        self.dimension = dim
        self.win.geometry(f"{dim[0]}x{dim[1]}")
        
        #setup & run process
        self.process = multiprocessing.Process(target=self.__run__)
        self.process.daemon = True

        #variables
        self.scale_var = tk.DoubleVar() 

        #initial gui setup
        label = tk.Label(self.win, text=win_name)
        label.grid(row=0, column=0, columnspan=1)

    #window functions for creation
    def __run__(self):
        self.win.mainloop()

    def start(self):
        self.process.start()
    
    def stop(self):
        self.process.terminate()

    #functions for element creation
    def createTrackbar(self, coords, label, length, onchange, range=[0, 100]):
        abs_coords = (coords[0], coords[1] + 1) 

        scale_text = tk.Label(self.win, text=label)
        scale = tk.Scale(self.win, variable = self.scale_var, from_ = range[0], to = range[1], orient = tk.HORIZONTAL, length=length, command= onchange)    
        scale_text.grid(row=abs_coords[1], column=abs_coords[0])
        scale.grid(row=abs_coords[1], column=abs_coords[0] + 1)
    
    def createButton(self, coords, label, onclick):
        abs_coords = (coords[0], coords[1] + 1)

        button = tk.Button(self.win, text = label, command = onclick)
        button.grid(row=abs_coords[1], column=abs_coords[0])
    
    def createLabel(self, coords, text):
        abs_coords = (coords[0], coords[1] + 1) 

        label = tk.Label(self.win, text=text)
        label.grid(row=abs_coords[1], column=abs_coords[0])

    def createViewbox(self, coords, img_source):
        abs_coords = (coords[0], coords[1] + 1) 

        #convert cv2 to PIL
        b,g,r = cv2.split(img_source)
        img = cv2.merge((r,g,b))
        img_pil = Image.fromarray(img)

        img_gui = ImageTk.PhotoImage(image=img_pil)

        viewbox = tk.Label(self.win, image=img_gui)
        viewbox.grid(row=abs_coords[1], column=abs_coords[0])

if __name__ == "__main__":
    img = cv2.imread('graph.png')
    print(img.shape)

    def Test1():
        print("your mom")

    def Test2(value):
        print("your dad", value)

    window = Window("Testing window", dim=(400, 400))

    window.createViewbox((0, 0), img)
    window.createTrackbar((0, 1), "Test", length=250, range=(0, 150), onchange=Test1)
    window.createLabel((0, 2), "Test dva")
    window.createButton((0, 3), "click me!", onclick=Test2)
    window.start()
    while True:
        pass
    window.stop()