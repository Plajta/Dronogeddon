import time
from djitellopy import Tello
from enum import Enum
from send import EmailSender
import cv2

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

sender = "plajta.corporation@hotmail.com"
reciver = "PlajtaCorp@proton.me"

mymail = EmailSender(sender,reciver)

margin = 5

class FlightController():
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller
        self.current_work_thread = 0

    def takeoff(self):
        takeoff = True

        with self.drone_controller.telloLock:
            try:
                tello.takeoff()
            except:
                takeoff = False
                print("\n\n--------------------------------------")
                print("takeoff not posible, check the batery")
                print("--------------------------------------\n\n")
                self.drone_controller.stop_program_and_land()
        return takeoff

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
            if in_distance > 0: 
                print(f"right by {abs(in_distance)}")
                speed = 50
            else:
                print(f"left by {abs(in_distance)}")
                speed = -50
            by = in_distance
        else:
            if out_distance < 0:
                print(f"left by {abs(out_distance)}")
                speed = -50
            else:
                print(f"right by {abs(out_distance)}")
                speed = 50
            by = out_distance

        print("speed: ",speed)
        print("rot")
        self.rotate(speed,end,deg,speed < 0,shift)
        
    def rotationBy(self,by):
        self.rotationTo(by+self.drone_controller.get_distance_data()[5])

    def rotate(self,speed,end,deg,direction,shift):
        time.sleep(0.1)
        tello.send_rc_control(0,0,0,speed)
        time.sleep(0.1)

        print(speed,end,deg,direction)
        
        while not((deg >= end) ^ direction):

            deg = (self.drone_controller.get_distance_data()[5])
            time.sleep(0.1)
            print(f"deg: {deg} deg-shift: {(deg + shift)%360} to tgD: {abs((deg+shift)%360-end)}  deg >= start: {(deg >= end) ^ direction} all: {((deg >= end) ^ direction)}")
            deg = (deg + shift)%360

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

    def see_and_send(self):
        pokracovac = True

        while pokracovac:
            data = self.drone_controller.get_intruder()
            if data:
                if data[0]:
                    print("je to tam moji hoši")
                    print(data[0])
                    cv2.imwrite("src/Flight_logs/img/img.jpg",data[1])
                    pokracovac = False
                    self.drone_controller.stop_program_and_land()
                    mymail.send_intruder_alert("src/Flight_logs/img/img.jpg")
                    print("poslano")
                else:
                    print("chyba na vašem příjmači")
                time.sleep(0.1)
            


    def flight_program(self):
        
        #takeoff = self.takeoff()   
        takeoff = True #čiste pro testování bez letu
        
        if takeoff:

            self.see_and_send()

            self.drone_controller.stop_program_and_land()

            print("end")


    