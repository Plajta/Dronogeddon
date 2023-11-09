import threading
import time
from djitellopy import Tello
from controler import DroneController
from telemetry import TelemetryScanner

tello = Tello()
tello.connect(False)

drone_controller = DroneController()

telemetry_scanner = TelemetryScanner(drone_controller)

print("fajabusa")

telemetry_thread = threading.Thread(target=telemetry_scanner.run)
telemetry_thread.start()

video_thread = threading.Thread(target=drone_controller.capture_video_with_telemetry)
video_thread.start()

print("ahoj")
time.sleep(10)
print("beboj")
drone_controller.stop_program = True
print("cechoj")

print(drone_controller.get_distance_data())





#flight_controller = FlightController(self)
#flight_controller.start_flight_program()


while True:
    pass
# self.stop_program_and_land()
# video_thread.join()