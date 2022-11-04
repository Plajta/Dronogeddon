import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# For webcam input:
def DetectPose(image):


  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = pose.process(image)

  # Draw the pose annotation on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
  return cv2.flip(image, 1), results