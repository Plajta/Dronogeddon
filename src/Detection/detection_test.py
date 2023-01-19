import cv2
import detection as det

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()

    detected = det.detect(frame)

    cv2.imshow("test", detected)
    key = cv2.waitKey(1)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()