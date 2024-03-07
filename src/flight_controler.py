import time
from djitellopy import Tello
from enum import Enum
from send import EmailSender
import cv2
import threading
import queue 

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


mymail = EmailSender()

margin = 5

class FlightControler():
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller
        self.stop_task = False
        self.human = False

    def takeoff(self):
        takeoff = True

        with self.drone_controller.telloLock:
            try:
                tello.send_rc_control(0,0,0,0)
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
            
            data.append([dis,deg])

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
            if self.stop_task:
                break
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

    def intruder_seeking(self):
        while not self.drone_controller.stop_program:
            data = self.drone_controller.get_intruder()
            if data:
                if data[0]:
                    print("je to tam moji hoši")
                    print(data[0])
                    cv2.imwrite("src/Flight_logs/img/img.jpg",data[1])
                    self.stop_task = True
                    self.human = True
                    try:
                        mymail.send_intruder_alert("src/Flight_logs/img/img.jpg")
                        print("poslano")
                    except:
                        print("neposlano")
                    break
                else:
                    print("chyba na vašem příjmači")
                time.sleep(0.1)
        return "intruder zaznamenan"

    def intruder_folowing(self):
        while  not self.drone_controller.stop_program:
            data = self.drone_controller.get_intruder()
            if data and data[0]:                                                                   #checks if there is a intruder entry
                intruder = data[0][0]                                                              #loads the first person if there is
                intruder = [round((intruder[0]+intruder[2])/2),round((intruder[1]+intruder[3])/2)] #calculate the midle point from the coordinates of rectangle
                distance_from_centre = intruder[0] - 960/2
                speed = round(distance_from_centre * 5/52)
                with self.drone_controller.telloLock:
                    tello.send_rc_control(0,0,0,speed)
                    print(speed,"\t|\t", distance_from_centre)

            

    def flight_program(self):
        program_buffer = {
            self.scan : []
        }


        takeoff = self.takeoff()   
        #takeoff = True #čiste pro testování bez letu
        
        if takeoff:
            intr = threading.Thread(target=self.intruder_seeking)
            #intr.start()

            # for instruction in program_buffer.keys():
            #     atributes = program_buffer[instruction]
            #     instruction(*atributes)

            #     if(self.human):
            #         self.intruder_folowing()

            with open("src/Flight_logs/scan_data/danovo_data_dva.txt","w") as f:
                data = self.scan()
                print(data)
                try:
                    f.write(f"{data}")
                except:
                    pass
                
            


            self.drone_controller.stop_program_and_land()

            print("end")
            intr.join
        else:
            self.drone_controller.stop_program()

    def run(self,queue,output_queue):
        while True:
            output = None
            step = queue.get()
            if step == "stop":
                break
            elif step is not None:
                print(step[0])
                output = step[0](*step[1])
            if not output == None:
                output_queue.put([step[0],output])


    