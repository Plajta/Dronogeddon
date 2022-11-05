import time, cv2

#from ThreadType import Thread_Inherit uselles
from threading import Thread
from queue import Queue #IMPORTANT

from djitellopy import Tello
from Mediapipe import *
from send import sendEmail

#variabe definitions
foceni = False
keepOperating = True
pixel_to_degree = 180/720
degree = 0
recorder = None
motor = None #passing empty threads
_sentinel = object() #an object that is going to stop the inter-thread communication

#global vars to change

tello = Tello()
tello.connect()

print("SLEEP TIMEOUT")
time.sleep(2)
print("SLEEP TIMEOUT END")

tello.streamon()
frame_read = tello.get_frame_read()

#thread definitions
def VideoRecorder(out_q):
    #          val dir
    Movement = [0, 0]

    #directions:
    #     1 UP
    #     |
    # 0 - - -2
    #     |
    #     3 DOWN

    height, width, _ = frame_read.frame.shape
    camera_center = [round(width/2), round(height/2)] #x, y format

    while keepOperating:
        image_arr = frame_read.frame
        image, results = DetectFace(image_arr)
        if results.detections:
            min_x = results.detections[0].location_data.relative_bounding_box.xmin
            min_y = results.detections[0].location_data.relative_bounding_box.ymin
            obj_width = results.detections[0].location_data.relative_bounding_box.width
            obj_height = results.detections[0].location_data.relative_bounding_box.height

            obj = [round(min_x*width), round(min_y*height), round(obj_width*width), round(obj_height*height)]
            #print(obj)
            #print(height, width)

            #just testing
            image_arr = cv2.rectangle(image_arr, (obj[0], obj[1]), (obj[0]+obj[2], obj[1]+obj[3]), (255, 0, 0), 4)

            #getting center of object
            obj_center = [obj[0]+round(obj_width/2), obj[1]+round(obj_height/2)]
            image_arr = cv2.circle(image_arr, (obj_center[0], obj_center[1]), radius=0, color=(255, 0, 0), thickness=5)

            distX = obj_center[0] - camera_center[0]
            degreeX = round(pixel_to_degree*distX)
            face_size = round(width/obj[2])

            print(degreeX)
            if degreeX>10:
                print("pravá")
                Movement = [round(degreeX/3), 2]
                out_q.put(Movement)
            elif degreeX<-10:
                print("levá")
                Movement = [round(degreeX/3), 2]
                out_q.put(Movement)
            else:
                print("stred")
                foceni = False
                print("mám tě čuráku")
                cv2.imwrite('imgs/img.jpg',image_arr)
                #Thread(target=sendEmail).start() lemme try something different
                sendEmail()
                Stop()
        else:
            Movement = [0, 0]
            out_q.put(Movement)

        cv2.imshow("drone", image_arr)
        cv2.imshow("drone_test", image)
        cv2.waitKey(1)

    out_q.put(_sentinel) #putting the end object to abort operation

def MotorControl(in_q):
    while keepOperating:
        data = in_q.get()

        if data is _sentinel:
            in_q.put(_sentinel)
            DestroyThreads()

        if data[1] == 0 and data[0] == 0:
            #start searching for pussies
            tello.move_forward(30)
            tello.rotate_clockwise(180)

        if data[1] == 2:
            tello.rotate_clockwise(data[0])
        elif data[1] == 0:
            tello.rotate_counter_clockwise(data[0])

def Startup():
    tello.takeoff()
    #add more if you need so

def Start():
    q = Queue #communication object
    recorder = Thread(target=VideoRecorder, args=(q))
    motor = Thread(target=MotorControl, args=(q, ))
    
    recorder.setDaemon(True) #trying some stuff with daemons
    motor.setDaemon(True)

    recorder.start()
    motor.start()
    

def Stop():
    keepRecording = False

def DestroyThreads():
    recorder.kill()
    motor.kill()
    exit(1)