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

    while keepRecording:
        image_arr = frame_read.frame
        image, results = DetectFace(image_arr)
        min_x = results.detections[0].location_data.relative_bounding_box.xmin
        min_y = results.detections[0].location_data.relative_bounding_box.ymin
        obj_width = results.detections[0].location_data.relative_bounding_box.width
        obj_height = results.detections[0].location_data.relative_bounding_box.height

        obj = [round(min_x*width), (min_y*height), (obj_width*width), (obj_height*height)]
        print(obj)

        #just testing
        cv2.rectangle(image, (min_x, min_y), (min_x+width, min_y+height), (255, 0, 0), 2)

        cv2.imshow("drone", image)
        cv2.waitKey(1)
        #cv2.destroyAllWindows() we dont need that

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