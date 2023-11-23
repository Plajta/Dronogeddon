import time
from djitellopy import Tello
from enum import Enum

tello = Tello()
tello.connect(False)

class Sensor(Enum):
    F = 0
    B = 3
    L = 1
    R = 2
    D = 4

class Direction(Enum):
    F = 1
    B = 1
    L = 0
    R = 0
    D = 2

margin = 5

class FlightController():
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller
        self.current_work_thread = 0

    def scan(self):
        tello.send_rc_control(0,0,0,0)
        data = []
        with self.drone_controller.telloLock:
            start = tello.get_yaw()
        deg = start
        tello.send_rc_control(0,0,0,30)

        neg = False
        while not(neg and deg >= start):
            print(f"neg: {neg} deg >= start: {deg >= start} all: {neg and deg >= start}")
            neg = deg < start
            print(deg >= start)

            with self.drone_controller.telloLock:
                deg = tello.get_yaw()

            dis = self.drone_controller.get_distance_data()
            print(f"vzdálenost: {dis[0]}  stupně: {deg}")
            
            data.append([dis[0],deg])

        tello.send_rc_control(0,0,0,0)
        return data
    
    def translation(self,sensor,target_value):
        direction = Direction[sensor].value
        sensor = Sensor[sensor].value
        distance = self.drone_controller.get_distance_data()[sensor]
        speed = [0,0,0,0]
        print(direction)
        directionist = ((sensor%2)*2-1)*-1
        stabilization = 0

        while (target_value < distance - margin or target_value > distance + margin) and stabilization < 100:
            if distance < target_value and speed[direction] != 40*directionist:
                speed[direction] = -40*directionist
                tello.send_rc_control(*speed)
                print("flight to -1")
                stabilization = 0
            elif distance > target_value and speed[direction] != 60*directionist:
                speed[direction] = 60*directionist
                tello.send_rc_control(*speed)
                print("flight to 1")
                stabilization = 0
            elif distance + margin > target_value > distance - margin:
                stabilization += 1
                tello.send_rc_control(0,0,0,0)

            distance = self.drone_controller.get_distance_data()[sensor]
            print(f"ano? {target_value < distance - margin or target_value > distance + margin} kolik: {distance} stabillization: {stabilization}")
            time.sleep(0.1)
        tello.send_rc_control(0,0,0,0)
        print(distance)

    def flight_program(self):
        with self.drone_controller.telloLock:
            tello.takeoff()
        self.current_work_thread = 1

        if self.current_work_thread == 0:
            print("start")
            scan_map = self.scan()
        elif self.current_work_thread == 1:
            print("start translation")
            self.translation("F",200)
            print("end translation")

        with self.drone_controller.telloLock:
            tello.land()

        print("end")

    