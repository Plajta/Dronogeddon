from djitellopy import Tello
from example import ToF as tf
import map_processing as vis
from threading import Thread
import threading
import cv2
from queue import Queue

zastavovac = threading.Event()
map_data = Queue()

def visualisation():
    vis.map_init()
    print(zastavovac.is_set())
    print(map_data.empty())
    while (not zastavovac.is_set()) or (not map_data.empty()):
        if map_data.empty() == False :
            mapdata = map_data.get()
            vis.update_map(*mapdata)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def scan():
    # visual = Thread(target=visualisation)
    # visual.start()
    tello = Tello()
    tello.connect(False)

    tello.send_rc_control(0,0,0,0)
    tello.takeoff()
    data = []



    start = tello.get_yaw()
    deg = start
    tello.send_rc_control(0,0,0,30)
    print("rotate")
    neg = False
    print(not(neg and deg >= start))
    print(start)
    while not(neg and deg >= start):
        print(f"neg: {neg} deg >= start: {deg >= start} all: {neg and deg >= start}")
        neg = deg < start
        print(deg >= start)

        deg = tello.get_yaw()
        dis = tf.mesurments()
        print(f"vzdálenost: {dis[0]}  stupně: {deg}")
        
        data.append([dis,deg])
        #map_data.put([dis[0],deg])

    zastavovac.set()
    tello.send_rc_control(0,0,0,0)
    print("konec")
    print(data)
    tello.land()



if __name__ == "__main__":
    scan()
    while not cv2.waitKey(0) & 0xFF == ord('q'):
        pass