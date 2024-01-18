from neural.pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER
from djitellopy import Tello
import cv2
import time
from AI import AImeter

tello = Tello()
tello.connect()
tello.streamon()
start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
font = cv2.FONT_HERSHEY_SIMPLEX
frame_read = tello.get_frame_read()
height, width, _ = frame_read.frame.shape
video_out = cv2.VideoWriter(f'src/Flight_logs/video/flight_log_{log_time}.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

ai = AImeter()
ai.start()

def video_recording_loop(data):
    img_out = frame_read.frame
    
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
            f"left: {data[1]} front: {data[0]} right: {data[2]}", 
            (10, 700), 
            font, 1/2, 
            (0, 255, 255), 
            2, 
            cv2.LINE_4) 
    
    ai.main(img_out)
    
    video_out.write(img_out)

    sleep(1/30)

    cv2.imshow("output_drone", img_out)

    if cv2.waitKey(1) == ord('q'):
        return False
    
    return True

def video_recording_finnish():
    cv2.destroyWindow("output_drone")
    video_out.release()
    ai.stop()
    tello.streamoff()

def sleep(time):
    start_time = time.monotonic()
    
    while True:
        current_time = time.monotonic()
        elapsed_time = current_time - start_time
        if elapsed_time >= time:
            break

if __name__ == "__main__":
    beh = True
    while beh:
        beh = video_recording_loop([0,0,0])
    video_recording_finnish()