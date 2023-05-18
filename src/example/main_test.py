import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
from djitellopy import Tello
import logging

from pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER

#tello = Tello()
#tello.LOGGER.setLevel(logging.DEBUG)

instructions = Queue()

frame_read = cv2.VideoCapture(0)
#tello.connect()
#tello.streamon()
#frame_read = tello.get_frame_read()

#functions
def Convert_to_Instructions(y_deviation, x_deviation, ob_area):
    #recompute to linear scale (lol)

    if x_deviation > 0.4:
        #compute rotation
        return ["rotate", x_deviation * 90]

    if ob_area > 0.8:
        #compute baackward shift
        return ["backward", 20]
    else:
        #compute forward shift
        return ["forward", ob_area * 200]

def stop_drone():
    #tello.land()

    frame_read.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def videoRecorder():
    ret, image = frame_read.read()
    torch_tensor = convert_to_tensor(image)

    output = model(torch_tensor)
    result = process_data(output, image)
    if len(result) != 0: #just when the data is avaliable
        y_dev, x_dev, area = compute_dev(result, image)

        y_dev = round(y_dev / SCREEN_CENTER[0], 2)
        x_dev = round(x_dev / SCREEN_CENTER[1], 2)

        print(y_dev, x_dev, area)

        instruction = Convert_to_Instructions(y_dev, x_dev, area)
        instructions.put(instruction)

    cv2.imshow("output_drone", image)
    if cv2.waitKey(1) == ord('q'):
        stop_drone()

def process_instructions():
    while True: #i hate this
        instruction = instructions.get()
        if len(instruction) == 0:
            continue

        if instruction[0] == "forward":
            print("forward")
        elif instruction[0] == "backward":
            print("backward")
        elif instruction[0] == "left":
            print("left")
        elif instruction[0] == "right":
            print("right")
        elif instruction[0] == "up-down":
            pass #TODO: dodělat!
        elif instruction[0] == "rotate":
            if instruction <= 0:
                #left
                print("rotate left")
            else:
                #right
                print("rotate right")
        
        else:
            pass

instructor = Thread(target=process_instructions)
instructor.start()

while True:
    videoRecorder()