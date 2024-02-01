from neural.pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER
import threading
import cv2
from time import sleep
import numpy as np

class AImeter:
    def __init__(self):
        self.pictureLock = threading.Lock()
        self.outputLock = threading.Lock()
        self.output = [0,0,0,0]
        self.picture = np.zeros((480, 640, 3), dtype=np.uint8)
        self.stop_program = False

    def get_picture(self):
        with self.pictureLock:
            return self.picture

    def set_picture(self, data):
        with self.pictureLock:
            self.picture = data.copy()

    def get_output(self):
        with self.outputLock:
            return self.output

    def set_output(self, data):
        with self.outputLock:
            self.output = data.copy()


    def AI_thread(self):
        while not self.stop_program:
            frame = self.get_picture()
            self.set_output(self.AI(frame))

    def AI(self,frame):
        """
        AI
        """
        torch_frame = convert_to_tensor(frame)
        out = model(torch_frame)
        coordinates = process_data(out, frame)

        return coordinates
    
    def start(self):
        ai_thread = threading.Thread(target=self.AI_thread)
        ai_thread.start()

    def stop(self):
        self.stop_program = True
    
    def main(self,input):
        self.set_picture(input)
        cords = self.get_output()
        for i in cords:
            cv2.rectangle(input, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 2)
            cv2.circle(input,(round((i[0]+i[2])/2),round((i[1]+i[3])/2)),25,(255,0,0),-1) 
        return cords



if __name__ == "__main__":
    vid = cv2.VideoCapture(0) 
    ai = AImeter()
    ai.start()
    cyklus = True
    #for i in range(100):
    while cyklus:
        ret, frame = vid.read() 
        cv2.imshow("input_drone", frame)
        ai.main(frame)
        cv2.imshow("output_AI", frame)
        if cv2.waitKey(1) == ord('q'):
            cyklus = False
    
    # while cyklus:
    #     ret, frame = vid.read() 
    #     cv2.imshow("output_drone", frame)
    #     cv2.imshow('frame', ai.AI(frame))
    #     if cv2.waitKey(1) == ord('q'):
            # cyklus = False

    ai.stop()
