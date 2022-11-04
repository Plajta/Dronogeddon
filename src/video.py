import time, cv2
from threading import Thread
from djitellopy import Tello
from Mediapipe import *

tello = Tello()

tello.connect()
time.sleep(2)
keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    height, width, _ = frame_read.frame.shape
    camera_center = [width/2, height/2] #x, y format

    while keepRecording:
        image_arr = frame_read.frame
        image, results = DetectFace(image_arr)
        if results.detections:
            min_x = results.detections[0].location_data.relative_bounding_box.xmin
            min_y = results.detections[0].location_data.relative_bounding_box.ymin
            obj_width = results.detections[0].location_data.relative_bounding_box.width
            obj_height = results.detections[0].location_data.relative_bounding_box.height

            obj = [round(min_x*width), round(min_y*height), round(obj_width*width), round(obj_height*height)]
            print(obj)
            print(height, width)

            #just testing
            image_arr = cv2.rectangle(image_arr, (int(min_x), int(min_y)), (int(min_x+obj_width), int(min_y+obj_height)), (255, 0, 0), 4)

            #getting center of object
            obj_center = [int(min_x+round(width/2)), int(min_y+round(height/2))]
            image_arr = cv2.circle(image_arr, (obj_center[0], obj_center[1]), radius=0, color=(255, 0, 0), thickness=5)

            

        cv2.imshow("drone", image_arr)
        cv2.imshow("drone_test", image)
        cv2.waitKey(1)

recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
#tello.move_up(100)
#tello.rotate_counter_clockwise(360)
#tello.flip_back()
#tello.flip_forward()
#tello.flip_left()
#tello.flip_right()
tello.land()

keepRecording = False
cv2.destroyAllWindows()
recorder.join()