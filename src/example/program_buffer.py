import queue
import threading
from time import sleep

def napocitej(kolik):
    for i in range(kolik):
        print(i+1)
        sleep(1)

def input_thread(queue):
    
    queue.put([napocitej,5])

def print_thread(queue):
    while True:
        # Block until an item is available in the queue
        step = queue.get()
        if step != None:
            step[0](step[1])
        

# Create a shared queue
shared_queue = queue.Queue()

# Create input and print threads
input_thread = threading.Thread(target=input_thread, args=(shared_queue,))
print_thread = threading.Thread(target=print_thread, args=(shared_queue,))

# Start the threads
input_thread.start()
print_thread.start()

# Wait for the input thread to finish (when the user types 'exit')
input_thread.join()

# Signal the print thread to finish
shared_queue.put(None)
print_thread.join()