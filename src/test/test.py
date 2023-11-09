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
        self.scanner = TelemetryScanner(self)
        

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
        self.scanner.stop()
        self.scanner.join()

    def start_flight(self):
        self.scanner.start()
        

class TelemetryScanner(threading.Thread,DroneController):
    def __init__(self, drone_controller):
        super().__init__()
        self.drone_controller = drone_controller
        self.stop_scanning = threading.Event()

    def run(self):
        while not self.stop_scanning.is_set():
            # Logic for scanning the area and updating distance data
            distance_data = self.mesurments()
            self.drone_controller.set_distance_data(distance_data)
            print("tel run")
            time.sleep(0.1)

    def mesurments(self):
        with self.telloLock:
            try:
                responses = tello.send_read_command('EXT tof?').split()
                if len(responses) < 6:
                    return self.mesurments()
                else:
                    return [int(responses[1]),int(responses[4]),int(responses[3]),int(responses[2]),int(responses[5])]
            except Exception as e:
                return self.mesurments()
            pass


    def stop(self):
        self.stop_scanning.set()

class FlightController():
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller

    def start_flight_program(self):
        while not self.drone_controller.stop_program:
            if self.current_work_thread == 0:  # Condition for scanning the area
                scan_map = self.scan()
                # Process data or take further action based on scan results
            elif self.current_work_thread != 0:  # Condition for other flight program steps
                pass  # Implement other flight steps

    def scan(self):
        tello.send_rc_control(0,0,0,0)
    
        data = []
        start = Tello.get_yaw()
        deg = start
        tello.send_rc_control(0,0,0,30)

        neg = False
        while not(neg and deg >= start):
            print(f"neg: {neg} deg >= start: {deg >= start} all: {neg and deg >= start}")
            neg = deg < start
            print(deg >= start)

            with self.telloLock:
                deg = tello.get_yaw()

            dis = self.get_distance_data()
            print(f"vzdálenost: {dis[0]}  stupně: {deg}")
            
            data.append([dis[0],deg])

        tello.send_rc_control(0,0,0,0)
        return data


if __name__ == "__main__":
    


    drone_controller = DroneController()
    drone_controller.start_flight()
    

    video_thread = threading.Thread(target=drone_controller.capture_video_with_telemetry)
    video_thread.start()

    #flight_controller = FlightController(self)
    #flight_controller.start_flight_program()

    
    while True:
        pass
    self.stop_program_and_land()
    video_thread.join()
