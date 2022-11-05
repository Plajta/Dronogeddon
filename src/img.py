import cv2
from threading import Thread
from djitellopy import Tello
from time import sleep
import numpy as np

tello = Tello()

tello.connect()

keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        #720 - height, 960- width
        image = frame_read.frame
        height = image.shape[0]
        width = image.shape[1]

        test = np.zeros((height, width, 3), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray , (13, 13), 0)

        thresh = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        addon_blur = cv2.blur(thresh, (5, 5))
        contours, hierarchy = cv2.findContours(addon_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(test, contours, -1, (0, 255, 0), 3)

        """ useless, develop something better
        delta_array = np.zeros((len(contours), 2), dtype=int)
        len_array = delta_array.copy()

        for iter, contour in enumerate(contours):
            if iter == 0:
                continue
            
            approx = cv2.approxPolyDP(contour,0.2 *cv2.arcLength(contour,True),True)
            if approx[0][0][0] - approx[1][0][0] == 0 or approx[0][0][1] - approx[0][0][1] == 0:
                continue

            cv2.drawContours(test, [approx], -1, (0,0,255), 3)
            
            dy = approx[1][0][0] - approx[0][0][0]
            dx = approx[1][0][1] - approx[0][0][1]
            delta_array[0] = round(dy/dx, 1)
            delta_array[1] = iter

            c = round(math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)), 1)
            len_array[0] = c
            len_array[1] = iter

        #sorting
        delta_array[delta_array[:, 0].argsort()]
        len_array[len_array[:, 0].argsort()]

        print(delta_array)
        """

        cv2.imshow("drone", test)
        cv2.imshow("main", thresh)
        cv2.waitKey(1)




recorder = Thread(target=videoRecorder)
recorder.start()

tello.takeoff()
tello.move_up(100)
tello.rotate_counter_clockwise(360)
tello.land()

keepRecording = False
recorder.join()