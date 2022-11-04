import time, cv2
from threading import Thread
from djitellopy import Tello
from Mediapipe import *

tello = Tello()

tello.connect()
time.sleep(2)
keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    height, width, _ = frame_read.frame.shape
    camera_center = [width/2, height/2] #x, y format

    while keepRecording:
        image_arr = frame_read.frame
        image, results = DetectFace(image_arr)
        cv2.imshow("drone", image)
        cv2.waitKey(1)
        #cv2.destroyAllWindows() we dont need that

recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
tello.move_up(100)
for i in range(3):
    tello.move_forward(250)
    tello.rotate_clockwise(90)
tello.move_forward(100)
tello.land()

keepRecording = False
cv2.destroyAllWindows()
recorder.join()