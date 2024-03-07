import threading
import queue 
from flight_controler import FlightControler
from time import sleep


def run(drone_controler):
    flight_controller = FlightControler(drone_controler)

    program_buffer = queue.Queue()
    program_output = queue.Queue()

    flight_controller_thread = threading.Thread(target=flight_controller.run,args=(program_buffer,program_output,))
    flight_controller_thread.start()


    # example of 
    # program_buffer.put([flight_controller.intruder_seeking,[]])
    # while not drone_controler.stop_program:
    #     huh = program_output.get()
    #     print(huh)
    #     print(huh[1] == "intruder zaznamenan")
    #     if huh[1] == "intruder zaznamenan":
    #         print("jes")
    #         drone_controler.stop_program_and_land()


    program_buffer.put("stop")
    flight_controller_thread.join()
    print(list(program_output.queue))