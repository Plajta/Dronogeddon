"""pole co dělá průměr posledních hodnot => počítat průměr 
	dopředu odstraní chybné methody
		zastavý provede další kontrolní měření - 5x"""
import cv2
from djitellopy import Tello
import ToF as tf
import time
from threading import Thread

side_margin_low = 800
side_margin_high = 1300
border_front = 50

start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
font = cv2.FONT_HERSHEY_SIMPLEX

data = [0,0,0,0,0,0]

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    # 创建一个VideoWrite对象，存储画面至./video.avi
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter(f'src/Flight_logs/video/flight_log_{log_time}.mkv', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        frame = frame_read.frame
        
        cv2.putText(frame, 
                f"{round(time.time()-start_time, 2)}s", 
                (10, 20), 
                font, 1/2, 
                (0, 255, 255),
                2,
                cv2.LINE_4) 
    
        cv2.putText(frame, 
                    f"{log_time}", 
                    (770, 20), 
                    font, 1/2, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4) 
        
        cv2.putText(frame, 
                f"{data}", 
                (10, 700), 
                font, 1/2, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4) 
        
        video.write(frame)
        time.sleep(1 / 30)

    video.release()

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


log_pad = open(f"src/Flight_logs/txt/flight_log_{log_time}.txt", 'w')
log_pad.write(f"start of the program at {log_time}\n\n")
log_pad.write(f"side margin low is: {side_margin_low} || side mrgin high is {side_margin_high} || border front is {border_front} || batery is n\n")

def log(text="", entr="\n"):
    log_pad.write(f"{round(time.time()-start_time, 2)}\t{text}{entr}")


tello = Tello()
tello.connect(False)
keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()
recorder = Thread(target=videoRecorder)
recorder.start()

log("tello takeoff")
tello.takeoff()
#tello.move_up(100)

pokracovac = True

while pokracovac:
    data = tf.mesurments()
    distanceFront = data[0]
    log(data)
    if distanceFront > border_front:
        tello.send_rc_control(0,30,0,0)
    else:
        tello.send_rc_control(0,0,0,0)
        pokracovac = False

while True:

    log("rotating left")

    tello.rotate_counter_clockwise(90)
    pokracovac = True
    time.sleep(0.5)

    log("go against wall")

    while pokracovac:
        data = distancemeter()
        

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
            speedFront = 20

        else:
            speedFront = 0
            pokracovac = False
            """
            tello.send_rc_control(0,0,0,0)
            for i in range(10):
                time.sleep(0.5)
                data = distancemeter()
                distanceFront = data[0]
                log(f"{i}. {data}")
            if distanceFront < border_front:
                speedFront = 0
                pokracovac = False
            else:
                log("planý poplach")"""

        tello.send_rc_control(speedSide,speedFront,0,0)

    
    


tello.send_rc_control(0,0,0,0)
tello.land
log_pad.close()
keepRecording = False
recorder.join()