import time, cv2
from threading import Thread
from queue import Queue #IMPORTANT

from djitellopy import Tello

#custom
#import detection

#vars
keepRecording = True

#Init
tello = Tello()
tello.connect()

print("SLEEP TIMEOUT")
time.sleep(2)
print("SLEEP TIMEOUT END")

tello.streamon()
frame_read = tello.get_frame_read()

def getBattery():
    return tello.get_battery()

def videoRecorder():
    height, width, _ = frame_read.frame.shape

    while keepRecording:
        #detection.Detect()
        print("RUN!")
        
recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
tello.move_up(100)
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
recorder.join()