import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue

from djitellopy import Tello
import threading

#custom
import Detection.detection as det
import Corner_Detection.Corner as Cor_det

def videoRecorder():
    while True:
        image = tello.get_frame_read().frame

        img_orig, coords = det.DetectPytorch(image)

        cv2.imshow("image", img_orig)
        cv2.waitKey(1)

def MotorController():
    while True:
        image = tello.get_frame_read().frame

        Cor_det.FindCorners()


tello = Tello()

tello.connect()

tello.streamon()

recorder = Thread(target=videoRecorder)
recorder.start()

"""
tello.takeoff()
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
recorder.join()
"""