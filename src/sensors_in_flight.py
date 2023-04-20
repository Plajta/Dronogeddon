from djitellopy import Tello
import ToF as tf
import time
from threading import Thread

border_low = 800
border_high = border_low + 150

tello = Tello()
tello.connect(False)

start_time = time.time()
log_time = time.strftime("%Y_%H_%M_%S", time.gmtime())
log_pad = open(f"flight_log_{log_time}.txt", 'a')
log_pad.write(f"start of the program at {log_time}\n\n")
log_pad.write(f"border low is: {border_low} || border high is {border_high} batery is\n\n")

def log(text=""):
    log_pad.write(f"{round(time.time()-start_time, 2)}: {text}\n")


log("tello takeoff")
tello.takeoff()

pokracovac = True
while pokracovac:
    data = tf.mesurments()
    distanceFront = data[0]
    log(data)
    if distanceFront < border_low:
        tello.send_rc_control(0,-20,0,0)
    elif distanceFront > border_high:
        tello.send_rc_control(0,20,0,0)
    else:
        tello.send_rc_control(0,0,0,0)
        pokracovac = False

while True:

    log("rotating left")

    tello.rotate_counter_clockwise(90)
    pokracovac = True
    time.sleep(0.5)

    log("go against wall")

    while pokracovac:
        data = tf.mesurments()
        distanceFront = data[0]
        distanceSide = data[2]
        log(data)
        speedFront = 0 
        speedSide = 0
        print(distanceSide)
        if distanceSide < border_low:
            speedSide = -20
        elif distanceSide > border_high:
            speedSide = 20
        else:
            speedSide = 0

        if distanceFront < border_low:
            speedFront = -20
        elif distanceFront > border_high:
            speedFront = 20
        else:
            speedFront = 0
            pokracovac = False

        tello.send_rc_control(speedSide,speedFront,0,0)

    
    


tello.send_rc_control(0,0,0,0)
tello.land
log_pad.close()