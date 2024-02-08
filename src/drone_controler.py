import threading
from djitellopy import Tello
import video as vid

tello = Tello()
tello.connect(False)

class DroneController:
    def __init__(self):
        self.distance_data = [0,0,0,0,0,0]
        self.lock = threading.Lock()
        self.telloLock = threading.Lock()
        self.intruder = threading.Lock()
        self.stop_program = False
        self.intruder_data = []

    def get_intruder(self):
        with self.intruder:
            return self.intruder_data

    def set_intruder(self, data):
        with self.intruder:
            self.intruder_data = data

    def get_distance_data(self):
        with self.lock:
            return self.distance_data

    def set_distance_data(self, data):
        with self.lock:
            self.distance_data = data

    def capture_video_with_telemetry(self):
        while not self.stop_program:
            distance = self.get_distance_data()
            beh,cords = vid.video_recording_loop(distance)
            self.set_intruder(cords)
            if beh:
                self.stop_program_and_land()
        vid.video_recording_finnish()
        with self.telloLock:
            try:
                tello.streamoff()
            except:
                print("can't cut the stream")

    def stop_program_and_land(self):
        self.stop_program = True
        with self.telloLock:
            tello.send_rc_control(0,0,0,0)
            try:
                tello.land()
            except:
                pass

