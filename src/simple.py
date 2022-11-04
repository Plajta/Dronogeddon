from djitellopy import Tello
from time import sleep
import logging
Tello.LOGGER.setLevel(logging.DEBUG)

tello = Tello()

tello.connect(False)
sleep(2)
tello.takeoff()

tello.move_left(100)
tello.rotate_clockwise(180)
tello.move_forward(100)

tello.land()