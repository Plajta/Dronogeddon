import tkinter as tk
from PIL import Image, ImageTk
import cv2
from multiprocessing import Queue, Process
import time

class Window:
    def __init__(self, win_name, loop_callback, dim=[700, 700]):
        self.win = tk.Tk()
        self.dimension = dim
        self.win.geometry(f"{dim[0]}x{dim[1]}")
        
        #setup & run process
        self.process = Process(target=self.__run__, args=(loop_callback, ))
        self.process.daemon = True

        #variables
        self.scale_var = tk.DoubleVar() 
        self.comm_queue = Queue()

        #initial gui setup
        label = tk.Label(self.win, text=win_name)
        label.grid(row=0, column=0, columnspan=1)

    #window functions for creation
    def __run__(self, callback):
        while True:
            if self.comm_queue.empty():
                callback()

            if self.comm_queue.get() == "stop":
                break

    def win_mainloop(self):
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.mainloop()
    
    def start(self):
        self.process.start()

    def close(self):
        self.win_closed = True
        self.win.destroy()
        self.comm_queue.put("stop")

    #functions for element creation
    def createTrackbar(self, coords, label, length, onchange, range=[0, 100]):
        abs_coords = (coords[0], coords[1] + 1) 

        scale_text = tk.Label(self.win, text=label)
        scale = tk.Scale(self.win, variable = self.scale_var, from_ = range[0], to = range[1], orient = tk.HORIZONTAL, length=length, command = onchange)    
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
        img_pil = img_pil.resize((400, 400))

        img_gui = ImageTk.PhotoImage(img_pil)

        viewbox = tk.Label(self.win)
        viewbox.grid(row=abs_coords[1], column=abs_coords[0])

        viewbox.configure(image=img_gui)
        viewbox.image = img_gui

        return viewbox

    def updateViewbox(self, viewbox, img_source):
        #convert cv2 to PIL
        b,g,r = cv2.split(img_source)
        img = cv2.merge((r,g,b))
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((400, 400))

        img_gui = ImageTk.PhotoImage(image=img_pil)

        viewbox.configure(image=img_gui)
        viewbox.image = img_gui

if __name__ == "__main__":
    img = cv2.imread('image.png')
    img2 = cv2.imread('image2.png')
    print(img.shape)

    def Test1(value):
        print("Test1", value)

    def Test2():
        print("Test2")

    def update():
        global designator
        time.sleep(1)
        if designator == "lenna1":
            window.updateViewbox(viewbox, img)
            designator = "lenna2"
        else:
            window.updateViewbox(viewbox, img2)
            designator = "lenna1"

    window = Window("Testing window", update, dim=(1000, 1000))
    global designator
    designator = "lenna1"
    viewbox = window.createViewbox((0, 0), img)
    window.createTrackbar((0, 1), "Test", length=250, range=(0, 150), onchange=Test1)
    window.createLabel((0, 2), "Test dva")
    window.createButton((0, 3), "click me!", onclick=Test2)
    window.start()
    window.win_mainloop() #TODO: fix this in more civilised manner