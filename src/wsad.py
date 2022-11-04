from djitellopy import Tello
import cv2, math, time
from time import sleep
import logging
Tello.LOGGER.setLevel(logging.DEBUG)
import keyboard

tello = Tello()
tello.connect(False)
sleep(2)


tello.streamon()


frame_read = tello.get_frame_read()

tello.takeoff()

while True:
    img = frame_read.frame
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff
    if keyboard.read_key() == 27: # ESC
        break
    elif keyboard.read_key() == 'w':
        tello.move_forward(30)
    elif keyboard.read_key() == 's':
        tello.move_back(30)
    elif keyboard.read_key() == 'a':
        tello.move_left(30)
    elif keyboard.read_key() == 'd':
        tello.move_right(30)
    elif keyboard.read_key() == 'e':
        tello.rotate_clockwise(30)
    elif keyboard.read_key() == 'q':
        tello.rotate_counter_clockwise(30)
    elif keyboard.read_key() == 'r':
        tello.move_up(30)
    elif keyboard.read_key() == ord('f'):
        tello.move_down(30)

tello.land()