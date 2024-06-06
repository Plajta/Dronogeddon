import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time

#main window class definition
class Window:
    def __init__(self, win_name, dim=[700, 700]):
        self.win = tk.Tk()
        self.dimension = dim
        self.win.geometry(f"{dim[0]}x{dim[1]}")

        #variables
        self.scale_var = tk.DoubleVar()
        self.thread_processes = []
        self.win_closed = False

        #initial gui setup
        label = tk.Label(self.win, text=win_name)
        label.grid(row=0, column=0, columnspan=1)

    def win_mainloop(self):
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.mainloop()

    def gui_function(self, target, delay):
        """
        Just a function to encapsulate target function so 
        you don't have to retype window checking every time.
        """

        while not self.win_closed:
            target()
            if delay != None:
                time.sleep(delay)

    def close(self):
        self.win_closed = True
        self.stop_threads()
        self.win.destroy()

    def set_thread(self, function, delay):
        thread = threading.Thread(target=self.gui_function, args=(function, delay))
        thread.daemon = True  # This will allow threads to exit when the main program exits
        self.thread_processes.append(thread)

    def run_threads(self):
        for thread in self.thread_processes:
            thread.start()

    def stop_threads(self):
        self.win_closed = True
        for thread in self.thread_processes:
            if thread.is_alive():
                # Threads should exit cleanly as win_closed is checked in the loop
                pass

    #functions for element creation
    def createTrackbar(self, coords, label, length, onchange, range=[0, 100]):
        abs_coords = (coords[0], coords[1] + 1)

        scale_text = tk.Label(self.win, text=label)
        scale = tk.Scale(self.win, variable=self.scale_var, from_=range[0], to=range[1], orient=tk.HORIZONTAL, length=length, command=onchange)
        scale_text.grid(row=abs_coords[1], column=abs_coords[0])
        scale.grid(row=abs_coords[1], column=abs_coords[0] + 1)

        return scale

    def createButton(self, coords, label, onclick):
        abs_coords = (coords[0], coords[1] + 1)

        button = tk.Button(self.win, text=label, command=onclick)
        button.grid(row=abs_coords[1], column=abs_coords[0])

        return button

    def createLabel(self, coords, text):
        abs_coords = (coords[0], coords[1] + 1)

        label = tk.Label(self.win, text=text)
        label.grid(row=abs_coords[1], column=abs_coords[0])

        return label

    def createViewbox(self, coords, end_coords, size, img_source="blank"):
        """
            Create viewbox for image to render
        """

        abs_coords = (coords[0], coords[1] + 1)

        if img_source == "blank":
            img_pil = Image.new("RGB", size)
        else:
            #convert cv2 to PIL
            b, g, r = cv2.split(img_source)
            img = cv2.merge((r, g, b))
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(size)

        img_gui = ImageTk.PhotoImage(img_pil)

        viewbox = tk.Label(self.win)
        viewbox.grid(row=abs_coords[1], column=abs_coords[0], rowspan=(end_coords[0] - coords[0]) + 1, columnspan=(end_coords[1] - coords[1]) + 1)

        viewbox.configure(image=img_gui)
        viewbox.image = img_gui

        return viewbox

    def updateViewbox(self, viewbox, viewbox_size, img_source):
        #convert cv2 to PIL
        b, g, r = cv2.split(img_source)
        img = cv2.merge((r, g, b))
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize(viewbox_size)

        img_gui = ImageTk.PhotoImage(image=img_pil)

        viewbox.configure(image=img_gui)
        viewbox.image = img_gui

#Examples definitons
class OpencvCamera:
    def __init__(self):
        vid = cv2.VideoCapture(0)
        y = vid.read()[1].shape[0]
        x = vid.read()[1].shape[1]
        viewbox_size = [x, y]

        def get_camera(): #just an example function
            ret, frame = vid.read()
            window.updateViewbox(viewbox, viewbox_size, frame)

        window = Window("Testing window", dim=(viewbox_size[0] + 100, 800))
        viewbox = window.createViewbox((0, 0), (0, 1), viewbox_size)
        window.createTrackbar((0, 1), "Test", length=250, range=(0, 150), onchange=lambda x: None)
        window.createLabel((0, 2), "Test dva")
        window.createButton((0, 3), "click me!", onclick=lambda: None)

        window.set_thread(get_camera, delay=None)
        window.run_threads()

        window.win_mainloop()

if __name__ == "__main__":
    OpencvCamera()