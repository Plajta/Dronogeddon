from djitellopy import Tello
import ToF as tf
import numpy as np
import time

tello = Tello()
tello.connect(False)

tello.send_rc_control(0,0,0,0)
tello.takeoff()
data = []

# for i in range(8):
#     deg = tello.get_yaw()
#     dis = tf.mesurments()
#     print(f"stupně: {deg[0]} vzdálenost {dis}")
#     tello.rotate_clockwise(45)
#     data.append([dis[0],deg])


start = tello.get_yaw()
deg = start
tello.send_rc_control(0,0,0,10)
print("rotate")
neg = False
print(not(neg and deg >= start))
print(start)
while not(neg and deg >= start):
    print(f"neg: {neg} deg >= start: {deg >= start} all: {neg and deg >= start}")
    neg = deg < start
    print(deg >= start)

    deg = tello.get_yaw()
    dis = tf.mesurments()
    print(f"vzdálenost: {dis[0]}  stupně: {deg}")
    data.append([dis[0],deg])
    
tello.send_rc_control(0,0,0,0)
print("konec")
print(data)
tello.land()

