from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import numpy as np
import torch
import cv2
import mediapipe as mp

#pytorch
CLASSES = 1 #we want to detect human being
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(weights=True, progress=True, pretrained_backbone=True).to(DEVICE)
MODEL.eval()

#mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

def DetectPytorch(img):
    img_orig = img.copy()

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to(DEVICE)
    detections = MODEL(image)[0]

    startX, startY, endX, endY = None, None, None, None

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i] * 100
        idx = int(detections["labels"][i])

        if idx == CLASSES and confidence > 90:
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES, confidence)
            print("[INFO] {}".format(label))

            cv2.rectangle(img_orig, (startX, startY), (endX, endY), (255, 0, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img_orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img_orig, (startY, startX, endY, endX)

def DetectMP(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
    return image, results
