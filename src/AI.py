import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_tensor = tf.convert_to_tensor(gray)

    detector_output = detector(image_tensor)
    print(detector_output["detection_classes"])

