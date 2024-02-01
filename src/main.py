from djitellopy import Tello
from drone_controler import DroneController
from telemetry import TelemetryScanner
from flight_controler import FlightController
import threading
import time
import keyboard

tello = Tello()
tello.connect(False)

def stop():
    drone_controller.stop_program = True
    telemetry_thread.join()
    video_thread.join

drone_controller = DroneController()
telemetry_scanner = TelemetryScanner(drone_controller)
telemetry_thread = threading.Thread(target=telemetry_scanner.run)
telemetry_thread.start()

video_thread = threading.Thread(target=drone_controller.capture_video_with_telemetry)
video_thread.start()

# flight_controller = FlightController(drone_controller)
# flight_controller.flight_program()


# stop()