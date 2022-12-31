# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import cv2
import numpy as np

import time
import os

def Test():
    img_pwd = "/mnt/Workspace/Projects/IntelligentHouseDrone/imgs/0.png"
    img = cv2.imread(img_pwd)
    img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)

    np_img = np.asarray(img)
    np_img = cv2.normalize(np_img.astype("float"), None, -0.5, 0.5, cv2.NORM_MINMAX)
    
    np_tensor = np.expand_dims(np_img, axis=0)

    print("ok4")

    #Apply image detector on a single image.
    #using efficientdet/d1 here, input = 640x640
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d1/1")
    detector_output = detector(np_tensor)
    class_ids = detector_output["detection_classes"]
    class_boxes = detector_output["detection_boxes"]
    class_scores = detector_output["detection_scores"]

    for i in range(class_ids.shape[1]):
        if int(class_ids[0][i]) == 1: #Person
            object = class_boxes[0][i] #x1 y1 x2 y2
            print("Score: " + str(class_scores[0][i]))

            x1 = object[0] * 640
            y1 = object[1] * 640
            x2 = object[2] * 640
            y2 = object[3] * 640

            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    cv2.imshow("Test", img)
    cv2.waitKey(0)



Test()