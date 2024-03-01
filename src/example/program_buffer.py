import threading
from time import sleep
import queue

class ProgramHandler():
    def __init__(self, queue):
        self.queue = queue

    def napocitej(self, kolik):
        # Example function
        for i in range(kolik):
            print(i + 1)
            sleep(1)

    def run(self):
        # Main function for the thread that listens for tasks in the queue and executes them
        while True:
            step = self.queue.get()
            if step == "stop":
                break
            elif step is not None:
                step[0](step[1])

# Create a shared queue
shared_queue = queue.Queue()

# Create an instance of the ProgramHandler class with the shared queue
PrH = ProgramHandler(shared_queue)

# Create a thread for the run method of ProgramHandler
input_thread = threading.Thread(target=PrH.run)

# Put a task in the queue (calling PrH.napocitej with argument 5)
shared_queue.put([PrH.napocitej, 5])

# Start the thread
input_thread.start()


#adding functions to queue...


# Put a stop signal in the queue to terminate the thread
shared_queue.put("stop")

# Wait for the input thread to finish (when "stop" is put into the queue)
input_thread.join()
