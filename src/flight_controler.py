import time
from djitellopy import Tello

tello = Tello()
tello.connect(False)

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
    
    def translation(axis,sensor):
        pass

    def flight_program(self):
        with self.drone_controller.telloLock:
            tello.takeoff()

        if self.current_work_thread == 0:
            print("start")
            scan_map = self.scan()
        elif self.current_work_thread != 0:
            pass 

        with self.drone_controller.telloLock:
            tello.land()

        print("end")

    