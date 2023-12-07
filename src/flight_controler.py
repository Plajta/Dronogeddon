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
        speed = 40 
        controlor = 40
        speeds = [0,0,0,0]
        print(direction)
        directionist = ((sensor%2)*2-1)*-1

        while (target_value < distance - margin or target_value > distance + margin):
            if distance < target_value and speeds[direction] != (-speed)*directionist:
                speeds[direction] = (-speed)*directionist
                tello.send_rc_control(*speeds)
                print("flight to -1")
            elif distance > target_value and speeds[direction] != speed*directionist:
                speeds[direction] = speed*directionist
                tello.send_rc_control(*speeds)
                print("flight to 1")

            if abs(distance-target_value) > 300 and speed != 80:
                speed = 80
                print("speed se to 80")
            elif 100 < abs(distance-target_value) < 300 and  speed != 40:
                speed = 40
                print("speed set to 40")
            elif abs(distance-target_value) < 100 and speed != 20:
                speed = 20
                print("speed set to 20")

            distance = self.drone_controller.get_distance_data()[sensor]
            print(f"ano? {target_value < distance - margin or target_value > distance + margin} kolik: {distance} to tgV {abs(distance-target_value)} speed:{speed}")
            time.sleep(0.1)

            if (target_value > distance - margin and target_value < distance + margin):
                time.sleep(0.5)

        tello.send_rc_control(0,0,0,0)
        print(distance)

    def rotationTo(self,end):

        

        tello.send_rc_control(0,0,0,0)
        deg = self.drone_controller.get_distance_data()[5]
        end = end%360

        shift = 180 - end
        end = 180
        deg = (deg+shift)%360
        print("rotating from: ",deg)
        print(F"rotating to {end}")

        in_distance = end - deg

        
        if end<deg:
            out_distance = 360 - deg +end
        else:
            out_distance = -(360- end +deg)


        print(deg)
        print(in_distance)
        print(out_distance)

        
        
        if abs(in_distance) < abs(out_distance):
            neg = True
            if in_distance > 0: 
                print(f"right by {abs(in_distance)}")
                speed = 30
            else:
                print(f"left by {abs(in_distance)}")
                speed = -30
            by = in_distance
        else:
            neg = False
            if out_distance < 0:
                print(f"left by {abs(out_distance)}")
                speed = -30
            else:
                print(f"right by {abs(out_distance)}")
                speed = 30
            by = out_distance

        print("speed: ",speed)
        print(neg)
        print("rot")
        self.rotate(speed,end,neg,deg,speed < 0,shift)
        

    def rotationBy(self):
        pass

    def rotate(self,speed,end,neg,deg,direction,shift):
        time.sleep(0.1)
        tello.send_rc_control(0,0,0,speed)
        time.sleep(0.1)

        print(speed,end,neg,deg,direction)
        
        while not(neg and (deg >= end) ^ direction):
            print(f"deg: {deg} to tgD: {abs(deg-end)} neg: {neg} deg >= start: {(deg >= end) ^ direction} all: {(neg and (deg >= end) ^ direction)}")
            if not(neg):
                neg = (deg < end) ^ direction

            deg = (self.drone_controller.get_distance_data()[5]+shift)%360
            time.sleep(0.1)
            

        tello.send_rc_control(0,0,0,0)
        

    def kruh(self):
        print("kruh")
        tello.send_rc_control(0,0,0,0)
        with self.drone_controller.telloLock:
            start = tello.get_yaw()
        deg = start
        tello.send_rc_control(0,40,0,60)
        time.sleep(0.1)
        neg = False
        while not(neg and deg >= start):
            neg = deg < start

            with self.drone_controller.telloLock:
                deg = tello.get_yaw()

            dis = self.drone_controller.get_distance_data()
            

        tello.send_rc_control(0,0,0,0)
            


    def flight_program(self):
        with self.drone_controller.telloLock:
            tello.takeoff()

        print(self.drone_controller.get_distance_data())
        time.sleep(2)
        # print("start translation")
        # self.translation("F",300)
        # print("end translation")

        by = 179
        deg = self.drone_controller.get_distance_data()[5]
        print(f"deg = {deg} rotating to {(deg+by)%360}")
        self.rotationTo(deg+by)
        time.sleep(2)
        print("done\n\n\n\n\n\n")
        deg = self.drone_controller.get_distance_data()[5]
        print(f"deg = {deg} rotating to {(deg+by)%360}")
        self.rotationTo(deg+by)
        print("done\n\n\n\n\n\n")
        by = -179
        deg = self.drone_controller.get_distance_data()[5]
        print(f"deg = {deg} rotating to {(deg+by)%360}")
        self.rotationTo(deg+by)
        print("done\n\n\n\n\n\n")
        deg = self.drone_controller.get_distance_data()[5]
        print(f"deg = {deg} rotating to {(deg+by)%360}")
        self.rotationTo(deg+by)

        #self.kruh()


        with self.drone_controller.telloLock:
            tello.land()

        print("end")
        print(self.drone_controller.get_distance_data()[5])

    