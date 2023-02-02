import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue #IMPORTANT

from djitellopy import Tello
import threading

#custom
import Detection.detection as det

#functions
def getBattery():
    return tello.get_battery()

def DroneController(q_stream, q_detection):
    while keepRecording:
        image = tello.get_frame_read().frame

        q_stream.put(image)
        image_det = q_detection.get()

        #plt.imshow(cv2.cvtColor(image, cv2.BGR2RGB))
        cv2.imshow("frame", image_det)
        cv2.waitKey(0)
        #plt.show()
        # print("stream running...")
    
    cv2.destroyAllWindows()

def ProcessImage(q_stream, q_detection):
    while keepComputing:
        image = q_stream.get()
        print("Processing image...")

        img_orig, coords = det.DetectMP(image)

        q_detection.put(img_orig)

def Video():
    while True:
        image = tello.get_frame_read().frame
        print("Processing image...")

        img_orig, coords = det.DetectMP(image)

        cv2.imshow("frame", img_orig)
        cv2.waitKey(0)

#vars
keepRecording = True
keepComputing = True

#Init
tello = Tello()
tello.connect()

print("SLEEP TIMEOUT")
time.sleep(2)
print("SLEEP TIMEOUT END")

tello.streamon()

q_stream = Queue() #communication object - stream input
q_det = Queue() #communication object - detection input

#controller = Thread(target=DroneController, args=(q_stream, q_det))
#computing = Thread(target=ProcessImage, args=(q_stream, q_det))
video = Thread(target=Video)

#controller.daemon = True
#computing.daemon = True

#controller.start()
#computing.start()
video.start()

tello.takeoff()
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
keepComputing = False

#controller.join()
#computing.join()
video.join()