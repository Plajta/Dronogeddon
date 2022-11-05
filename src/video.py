import time, cv2
from threading import Thread
from djitellopy import Tello
from Mediapipe import *
from send import sendEmail
from main import battery

tello = Tello()

tello.connect()
time.sleep(2)

foceni = False
keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()
#code definitions
pixel_to_degree = 180/720
degree = 0
recorder = None

def videoRecorder():
    height, width, _ = frame_read.frame.shape
    camera_center = [round(width/2), round(height/2)] #x, y format

    while keepRecording:
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
                tello.rotate_clockwise(round(degreeX/3))
            elif degreeX<-10:
                print("levá")
                tello.rotate_counter_clockwise(abs(round(degreeX/3)))
            else:
                print("stred")
                foceni = False
                print("mám tě čuráku")
                cv2.imwrite('imgs/img.jpg',image_arr)
                Thread(target=sendEmail).start()
                Stop()


        cv2.imshow("drone", image_arr)
        cv2.imshow("drone_test", image)
        cv2.waitKey(1)

def Start():

    if tello.get_battery() <= 20:
        print("Low battery!")
        return;

    recorder = Thread(target=videoRecorder)
    recorder.start()

    tello.takeoff()

    #tello.rotate_counter_clockwise(360)
    #tello.flip_back()
    #tello.flip_forward()
    #tello.flip_left()
    #tello.flip_right()
    for i in range(3):
        tello.rotate_clockwise(180)
        tello.move_forward(30)



def Stop():
    keepRecording = False
    tello.land()
    cv2.destroyAllWindows()
    recorder.join()

battery = tello.get_battery()