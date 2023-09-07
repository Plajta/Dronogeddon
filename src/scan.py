from djitellopy import Tello
import ToF as tf
import numpy as np

tello = Tello()
tello.connect(False)

tello.takeoff()
data = []

# for i in range(8):
#     deg = tello.get_yaw()
#     dis = tf.mesurments()
#     print(f"stupně: {deg[0]} vzdálenost {dis}")
#     tello.rotate_clockwise(45)
#     data.append([dis[0],deg])


start = tello.get_yaw()
tello.send_rc_control(0,0,0,10)
while True:
    deg = tello.get_yaw()
    dis = tf.mesurments()
    print(f"stupně: {deg[0]} vzdálenost {dis}")
    data.append([dis[0],deg])
tello.send_rc_control(0,0,0,0)

print(data)
tello.land()

