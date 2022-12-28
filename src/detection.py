# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import cv2
import numpy as np

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import time
import os

def Test():
    img = cv2.imread(os.getcwd().replace("src/detection.py", "") + "/imgs/0.png")
    img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)

    np_img = np.asarray(img)
    np_img = cv2.normalize(np_img.astype("float"), None, -0.5, 0.5, cv2.NORM_MINMAX)
    
    np_tensor = np.expand_dims(np_img, axis=0)

    #Apply image detector on a single image.
    #using efficientdet/d1 here, input = 640x640
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d1/1")
    detector_output = detector(np_tensor)
    class_ids = detector_output["detection_classes"]

    print(class_ids)


Test()