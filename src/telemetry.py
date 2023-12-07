import threading
import time
from djitellopy import Tello

tello = Tello()
tello.connect(False)

class TelemetryScanner(threading.Thread,):
    def __init__(self, drone_controller):
        super().__init__()
        self.drone_controller = drone_controller
        self.stop_scanning = threading.Event()

    def run(self):
        while not self.drone_controller.stop_program:
            # Logic for scanning the area and updating distance data
            distance_data = self.mesurments()
            self.drone_controller.set_distance_data(distance_data)
            time.sleep(0.1)

    def mesurments(self):
        with self.drone_controller.telloLock:
            try:
                responses = tello.send_read_command('EXT tof?').split()
                if len(responses) < 6:
                    return self.mesurments()
                else:
                    return [int(responses[1]),int(responses[4]),int(responses[3]),int(responses[2]),int(responses[5]),tello.get_yaw()+180]
            except Exception as e:
                return self.mesurments()
            pass


    def stop(self):
        self.stop_scanning.set()

if __name__ == "__main__":
    tello.send_read_command('EXT tof?')