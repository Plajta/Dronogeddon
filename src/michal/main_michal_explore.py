import cv2
#from neural.pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER
from djitellopy import Tello
import time
from threading import Thread
import threading
from queue import Queue

telloLock = threading.Lock()

side_margin_low = 100
side_margin_high = 150
side_margin_ignore = 180
border_front_low = 100
border_front_high = 200
SPEED_LOW = 20
SPEED_HI = 50

start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
font = cv2.FONT_HERSHEY_SIMPLEX

zastavovac = threading.Event()

video_out = None

mean = [[],[],[]]

tello = Tello()

current_work_thread = None

instructions_cam = Queue()
telemetri = Queue()
pictures = Queue()
measurements_for_instructor = Queue()
measurements_for_ui = Queue()
measurements_for_map = Queue()

def mesurments():
    with telloLock:
        try:
            responses = tello.send_read_command('EXT tof?').split()
            if len(responses) < 6:
                return mesurments()
            else:
                return [int(responses[1]),int(responses[4]),int(responses[3]),int(responses[2]),int(responses[5])]
        except Exception as e:
            return mesurments()
        pass

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
    print("stop")
    zastavovac.set()
    with telloLock:
        tello.send_rc_control(0,0,0,0)
        tello.land()
        video_out.release()
        print("video relesed")
        instructor_thread.join()
        # AImeter.join()
        tof_thread.join()
        if current_work_thread != None:
            current_work_thread.join()
        tello.streamoff()   
        pass
    print("stream off")
    log_pad.close()
    print("logpad ")    
    exit(1)

def video_recording():
    frame_read = tello.get_frame_read()
    height, width, _ = frame_read.frame.shape
    video_out = cv2.VideoWriter(f'src/Flight_logs/video/flight_log_{log_time}.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    mesurment = None
    while not measurements_for_ui.empty():
        mesurment = measurements_for_ui.get()

    data = [0,0,mesurment]
    while not zastavovac.is_set():
        img_out = frame_read.frame

        if telemetri.empty() == False :
            data = telemetri.get()
            print("video:")
            print(data)


        cv2.putText(img_out, 
                    f"{round(time.time()-start_time, 2)}s", 
                    (10, 20), 
                    font, 1/2, 
                    (0, 255, 255),
                    2,
                    cv2.LINE_4) 
        
        cv2.putText(img_out, 
                    f"{log_time}", 
                    (770, 20), 
                    font, 1/2, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4) 
        
        cv2.putText(img_out, 
                f"left: {data[2][1]} front: {data[2][0]} right: {data[2][2]}", 
                (10, 700), 
                font, 1/2, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4) 
                
        video_out.write(img_out)
        pictures.queue.clear()
        pictures.put(img_out)
        
        time.sleep(1 / 30)

        cv2.imshow("output_drone", img_out)
        if cv2.waitKey(1) == ord('q'):
            stop_drone()


def AI():
    while not zastavovac.is_set():
        """
        AI
        """
        if pictures.empty() == False :

            image = pictures.get()


            torch_tensor = convert_to_tensor(image)

            output = model(torch_tensor)
            result = process_data(output, image)
            cv2.rectangle(image, (result[0],result[1]), (result[2],result[3]), (0, 255, 0), 2)
            print(result)
            cv2.imshow("output_drone", image)

def process_ToF():
    while not zastavovac.is_set():
        data = mesurments()
        measurements_for_instructor.put(data)
        measurements_for_ui.put(data)
        time.sleep(0.1)

def select_next_step():
    next_step = None

    if current_work_thread == None:
        next_step = scan_in_cyrcle
    else:
        stop_drone()
        return
    
    current_work_thread = Thread(target=next_step)
    current_work_thread.start()

def scan_in_cyrcle():
    yaw_start = tello.get_yaw() + 180
    yaw_rotated = 0
    yaw_current = yaw_start

    tello.send_rc_control(0,0,0,10)
    print("rotate")

    while yaw_rotated <= 360 :
        
        yaw_prew = yaw_current
        yaw_current = tello.get_yaw() + 180
        yaw_rotated += abs(yaw_current - yaw_start)

        didistances = tf.mesurments()
        print(f"vzdálenost: {didistances}  stupně: {yaw_current}")
        measurements_for_map.append([didistances, yaw_current])
        
    tello.send_rc_control(0,0,0,0)

def process_instructions():
    while not zastavovac.is_set():
        try:
            process_instruction_unsafe()
        except Exception as e:
            stop_drone()
            break

def process_instruction_unsafe():
    if instructions_cam.empty() == False :
        instruction_cam = instructions_cam.get()

        print("OUT CAM:")
        print(instruction_cam)

    if measurements_for_instructor.empty() == False :
        data = None
        while not measurements_for_instructor.empty():
            data = measurements_for_instructor.get()
        
        distance_front = data[0]
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

        if distance_sideR > side_margin_ignore:
            speedSide = 0
        elif distance_sideR < local_side_margin_low:
            speedSide = -20
        elif distance_sideR > local_side_margin_high:
            speedSide = 20
        else:
            speedSide = 0

        # kdyz laser propali okna
        if distance_front > 1200:
            with telloLock:
                tello.send_rc_control(0,0,0,0)
                tello.rotate_counter_clockwise(10)
                pass

        if distance_front > border_front_high:
            speedFront = SPEED_HI
        elif distance_front > border_front_low:
            speedFront = SPEED_LOW
        else:
            with telloLock:
                tello.send_rc_control(0,0,0,0)
                tello.rotate_counter_clockwise(90)
                pass

        with telloLock:
            tello.send_rc_control(speedSide,speedFront,0,0)
            time.sleep(0.2)
            pass


log_pad = open(f"src/Flight_logs/txt/flight_log_{log_time}.txt", 'w')
log_pad.write(f"start of the program at {log_time}\n\n")
log_pad.write(f"side margin low is: {side_margin_low} || side mrgin high is {side_margin_high} || border front low is {border_front_low} || batery is n\n")

def log(text="", entr="\n"):
    log_pad.write(f"{round(time.time()-start_time, 2)}\t{text}{entr}")

instructor_thread = Thread(target=process_instructions)
tof_thread = Thread(target=process_ToF)
#AImeter = Thread(target=AI)
#videoRecorder = Thread(target=video_recording)

tello.connect(False)
tello.streamon()
#AImeter.start()
#videoRecorder.start()

tello.takeoff()
log("tello takeoff")

time.sleep(2)

tof_thread.start()
select_next_step()
