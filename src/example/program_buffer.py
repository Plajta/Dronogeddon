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

    def vypis(self,string):
        for i in string:
            print(i,end=" ")
            sleep(1)
        print()

    def jedna_ples(self,vstup):
        print(f"jedna plus {vstup} se rovná")
        sleep(1)
        print(1+vstup)
        return vstup
    
    def printd(self):
        print("ahoj")

    def run(self,queue,output_queue):
        # Main function for the thread that listens for tasks in the queue and executes them
        while True:
            output = None
            step = queue.get()
            if step == "stop":
                break
            elif step is not None:
                output = step[0](*step[1])
            if not output == None:
                output_queue.put([step[0],output])

# Create a shared queue
shared_queue = queue.Queue()
output_queue = queue.Queue()

# Create an instance of the ProgramHandler class with the shared queue
PrH = ProgramHandler(shared_queue)

# Create a thread for the run method of ProgramHandler
input_thread = threading.Thread(target=PrH.run,args=(shared_queue,output_queue,))

# Put a task in the queue (calling PrH.napocitej with argument 5)
shared_queue.put([PrH.printd,[]])
shared_queue.put([PrH.jedna_ples,2])
shared_queue.put([PrH.napocitej, 5])

# Start the thread
input_thread.start()

shared_queue.put([PrH.vypis, "ahoj"])
#adding functions to queue...


# Put a stop signal in the queue to terminate the thread
shared_queue.put("stop")

# Wait for the input thread to finish (when "stop" is put into the queue)
input_thread.join()

#print all outputs
print(list(output_queue.queue))
print(PrH.jedna_ples)
