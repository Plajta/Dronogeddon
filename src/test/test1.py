import time
from enum import Enum



class Sensor(Enum):
    F = 1
    B = 2
    L = 3
    R = 4
    D = 5

class Direction(Enum):
    F = 1
    B = 1
    L = 0
    R = 0
    D = 2

class Tello():
    def send_rc_control(self,s,f,d,y):
        print(f"side: {s} front: {f} fown: {d} yaw: {y}")

tello = Tello()
start = 4000
tv = 500
margin = 40

def translation(sensor,target_value):
    direction = Direction[sensor].value
    sensor = Sensor[sensor].value
    distance = start
    speed = [0,0,0,0]
    print(direction)


    while target_value < distance - margin or target_value > distance + margin:
        if distance > target_value and speed[direction] != -40:
            speed[direction] = -40
            tello.send_rc_control(*speed)
            print("flight to -1")
        elif distance < target_value and speed[direction] != 40:
            speed[direction] = 40
            tello.send_rc_control(*speed)
            print("flight to 1")
        distance += speed[direction]
    tello.send_rc_control(0,0,0,0)
    print(distance)

translation("F",tv)
