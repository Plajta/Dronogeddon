import time
from threading import Thread
from djitellopy import Tello

tello = Tello()
time.sleep(5)
tello.connect(False)
time.sleep(5)
tello.takeoff()
tello.rotate_counter_clockwise(90)


while(True):
    response = tello.send_read_command('EXT tof?')
    distance = int(response[4:])
    print('Distance is: ' + str(distance))
    if distance < 500:
        print('Rotating ...')
        tello.rotate_counter_clockwise(90)
    print('Waiting ...')
    time.sleep(1)
    