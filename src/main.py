from djitellopy import Tello
from drone_controler import DroneController
from telemetry import TelemetryScanner
from flight_controler import FlightControler
import threading
import path_processing

tello = Tello()
tello.connect(False)

def stop():
    drone_controler.stop_program = True
    telemetry_thread.join()
    video_thread.join

drone_controler = DroneController()
telemetry_scanner = TelemetryScanner(drone_controler)
telemetry_thread = threading.Thread(target=telemetry_scanner.run)
telemetry_thread.start()

video_thread = threading.Thread(target=drone_controler.capture_video_with_telemetry)
video_thread.start()

# flight_controller = FlightControler(drone_controler)
# flight_controller.flight_program()

path_processing.run(drone_controler)


print("stop")
stop()
print("všechno je to uděláno všechno je to hotovo")