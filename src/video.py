import time, cv2
from threading import Thread
from djitellopy import Tello
from Mediapipe import *

tello = Tello()

tello.connect()

keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    height, width, _ = frame_read.frame.shape

    while keepRecording:
        image_arr = frame_read.frame
        image, results = DetectFaces(image_arr)
        #time.sleep(1 / 30)
        cv2.imshow("drone", image)
        cv2.waitKey(1)
        #cv2.destroyAllWindows() we dont need that

recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
tello.move_up(100)
tello.rotate_counter_clockwise(360)
tello.flip_back();
tello.flip_forward();
tello.flip_left();
tello.flip_right();
tello.land()

keepRecording = False
cv2.destroyAllWindows()
recorder.join()