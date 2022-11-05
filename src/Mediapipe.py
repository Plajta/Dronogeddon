import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

# For webcam input:
def DetectFace(image):


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
