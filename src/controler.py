import threading
import time
from djitellopy import Tello
import video as vid

tello = Tello()
tello.connect(False)

class DroneController:
    def __init__(self):
        self.distance_data = [0,0,0]
        self.lock = threading.Lock()
        self.telloLock = threading.Lock()
        self.stop_program = False
        self.current_work_thread = 0
        

    def get_distance_data(self):
        with self.lock:
            return self.distance_data

    def set_distance_data(self, data):
        with self.lock:
            self.distance_data = data

    def capture_video_with_telemetry(self):
        while not self.stop_program:
            distance = self.get_distance_data()
            beh = vid.video_recording_loop(distance)
        vid.video_recording_finnish()

    def stop_program_and_land(self):
        self.stop_program = True
        with self.telloLock:
            tello.send_rc_control(0,0,0,0)
            tello.land()

