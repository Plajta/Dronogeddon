#-------------------------------------
#NEFUNGUJE
#-------------------------------------

import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time
import cv2

# Apply image detector on a single image.

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

def detect(img):
    img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)

    np_img = np.asarray(img)
    np_img = cv2.normalize(np_img.astype("float"), None, -0.5, 0.5, cv2.NORM_MINMAX)

    np_tensor = np.expand_dims(np_img, axis=0)

    detector_output = detector(np_tensor)
    class_ids = detector_output["detection_classes"]

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