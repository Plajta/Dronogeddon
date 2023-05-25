"""pole co dělá průměr posledních hodnot => počítat průměr 
	dopředu odstraní chybné methody
		zastavý provede další kontrolní měření - 5x"""
import cv2
from djitellopy import Tello
import ToF as tf
import time
from threading import Thread
from queue import Queue

from neural.pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER

side_margin_low = 800
side_margin_high = 1300
border_front = 500

start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
font = cv2.FONT_HERSHEY_SIMPLEX

pokracovac = True
zastavovac = False



mean = [[],[],[]]
def distancemeter():
    data = tf.mesurments()
    output = data
    for i in range(0,3):
        mean[i].append(data[i])

    if len(mean[0]) > 13:
        for i in range(0,3):
            mean[i].pop(0)
    for i,j in enumerate(mean):
        output.append(0)
        for k in j:
            output[i+3] += k
        output[i+3] = round(output[i+3] / len(j))

    return(output)

#functions
def Convert_to_Instructions(y_deviation, x_deviation, ob_area):
    #recompute to linear scale (lol)

    if abs(x_deviation) > 0.25:
        #compute rotation
        in1 = ["rotate", int(round(x_deviation * 45))]
    else:
        in1 = []

    if ob_area > 0.8:
        #compute baackward shift
        in2 = ["backward", 50]
    else:
        #compute forward shift
        in2 = ["forward", 20 if int(round(ob_area * 100)) <= 20 else int(round(ob_area * 100))]

    return [in2, in1]

def stop_drone():
    tello.land()
    tello.streamoff()
    instructor.join()
    zastavovac = True
    exit(1)

def videoRecorder():
    image = frame_read.frame

    data = distancemeter()

    """
    CAMERA  
    """
    cv2.putText(image, 
                f"{round(time.time()-start_time, 2)}s", 
                (10, 20), 
                font, 1/2, 
                (0, 255, 255),
                2,
                cv2.LINE_4) 
    
    cv2.putText(image, 
                f"{log_time}", 
                (770, 20), 
                font, 1/2, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4) 
    
    cv2.putText(image, 
            f"{data}", 
            (10, 700), 
            font, 1/2, 
            (255, 255, 255), 
            2, 
            cv2.LINE_4) 
    
    torch_tensor = convert_to_tensor(image)

    output = model(torch_tensor)
    result = process_data(output, image)
    if len(result) != 0: #just when the data is avaliable
        y_dev, x_dev, area = compute_dev(result, image)

        y_dev = round(y_dev / SCREEN_CENTER[0], 2)
        x_dev = round(x_dev / SCREEN_CENTER[1], 2)

        instruction = Convert_to_Instructions(y_dev, x_dev, area)
        #print("IN:")
        #print(instruction)

        instructions_cam.put(instruction)

    """
    ToF
    """

    
        

    distance_front = data[3]
    distance_sideL = data[1]
    distance_sideR = data[2]
    speedFront = 0 
    speedSide = 0


    if distance_sideL + distance_sideR < side_margin_high * 2:
        local_side_margin_high = (distance_sideL + distance_sideR)/2
        local_side_margin_low = local_side_margin_high - 300
        log(f"{data} || soucet:{distance_sideL + distance_sideR} True H:{local_side_margin_high} L:{local_side_margin_low}")

    else:
        local_side_margin_high = side_margin_high
        local_side_margin_low = side_margin_low
        log(f"{data} || soucet:{distance_sideL + distance_sideR} False H:{local_side_margin_high} L:{local_side_margin_low}")

    if distance_sideR < local_side_margin_low:
        speedSide = -20

    elif distance_sideR > local_side_margin_high:
        speedSide = 20

    else:
        speedSide = 0


    if distance_front > border_front:
        speedFront = 30

    else:
        speedFront = 0
        tello.rotate_counter_clockwise(90)

    instructions_ToF.put([speedSide, speedFront])
    #tello.send_rc_control(speedSide,speedFront,0,0)
    return image

def process_instructions():
    while zastavovac == False: #i hate this
        """
        CAMERA
        """
        if instructions_cam.empty() == False or instructions_ToF == False :
            instruction_cam = instructions_cam.get()

            instructions_cam.queue.clear()

            print(2)

            instruction_ToF = instructions_ToF.get()
            instructions_ToF.queue.clear()

            print(3)

            print("OUT CAM:")
            print(instruction_cam)
            print("OUT ToF:")
            print(instruction_ToF)

            """
            ToF
            """

            print(instruction_ToF[0],instruction_ToF[1],0,0)
            tello.send_rc_control(instruction_ToF[0],instruction_ToF[1],0,0)


log_pad = open(f"src/Flight_logs/txt/flight_log_{log_time}.txt", 'w')
log_pad.write(f"start of the program at {log_time}\n\n")
log_pad.write(f"side margin low is: {side_margin_low} || side mrgin high is {side_margin_high} || border front is {border_front} || batery is n\n")

def log(text="", entr="\n"):
    log_pad.write(f"{round(time.time()-start_time, 2)}\t{text}{entr}")


tello = Tello()
instructions_cam = Queue()
instructions_ToF = Queue()

tello.connect(False)
keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()
instructor = Thread(target=process_instructions)



log("tello takeoff")
tello.takeoff()
#tello.move_up(100)

while pokracovac:
    data = tf.mesurments()
    distanceFront = data[0]
    log(data)
    if distanceFront > border_front:
        tello.send_rc_control(0,20,0,0)
    else:
        tello.send_rc_control(0,0,0,0)
        pokracovac = False

video_out = cv2.VideoWriter(f'src/Flight_logs/video/flight_log_{log_time}.mkv', cv2.VideoWriter_fourcc(*'XVID'), 30, (480, 640)) #TODO: check if correct

img_out = videoRecorder()
instructor.start()

while True:
    img_out = videoRecorder()
    video_out.write(img_out)
    time.sleep(1 / 30)

    cv2.imshow("output_drone", img_out)
    if cv2.waitKey(1) == ord('q'):
        stop_drone()