import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue #IMPORTANT

from djitellopy import Tello
import threading

#custom
import Detection.detection as det

def videoRecorder():
    while True:
        image = tello.get_frame_read().frame

        img_orig, coords = det.DetectPytorch(image)

        cv2.imshow("image", img_orig)
        cv2.waitKey(1)

tello = Tello()

tello.connect()

tello.streamon()

recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
recorder.join()