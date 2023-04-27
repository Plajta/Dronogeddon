"""pole co dělá průměr posledních hodnot => počítat průměr 
	dopředu odstraní chybné methody
		zastavý provede další kontrolní měření - 5x"""

from djitellopy import Tello
import ToF as tf
import time
from threading import Thread

margin_side_low = 800
margin_side_high = 1500
border_front = 950

mean = [[],[],[]]

def distancemeter():
    data = tf.mesurments()
    output = data
    for i in range(0,3):
        mean[i].append(data[i])

    if len(mean[0]) > 10:
        for i in range(0,3):
            mean[i].pop(0)
    for i,j in enumerate(mean):
        output.append(0)
        for k in j:
            output[i+3] += k
        output[i+3] /= len(j)

    return(output)





start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
log_pad = open(f"src/Flight_logs/flight_log_{log_time}.txt", 'w')
log_pad.write(f"start of the program at {log_time}\n\n")
log_pad.write(f"side margin low is: {margin_side_low} || side mrgin high is {margin_side_high} || border front is {border_front}\n\n")

def log(text=""):
    log_pad.write(f"{round(time.time()-start_time, 2)}: {text}\n")

tello = Tello()
tello.connect(False)

log("tello takeoff")
tello.takeoff()
tello.move_up(100)

pokracovac = True
while pokracovac:
    data = tf.mesurments()
    distanceFront = data[0]
    log(data)
    if distanceFront > border_front:
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
        data = distancemeter()
        log(data)

        distanceFront = data[0]
        distanceSide = data[2]
        speedFront = 0 
        speedSide = 0

        if distanceSide < margin_side_low:
            speedSide = -20

        elif distanceSide > margin_side_high:
            speedSide = 20

        else:
            speedSide = 0


        if distanceFront > border_front:
            speedFront = 20

        else:
            speedFront = 0
            pokracovac = False

        tello.send_rc_control(speedSide,speedFront,0,0)

    
    


tello.send_rc_control(0,0,0,0)
tello.land
log_pad.close()