import cv2
import numpy as np
from read_dataset import test

def main(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,0,200)

    cv2.imshow("result", edges)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    return edges

for X, y in test:
    for img_torch in X:
        
        R = img_torch[0].numpy()
        G = img_torch[1].numpy()
        B = img_torch[2].numpy()

        img = np.asarray(cv2.merge((B, G, R)) * 255, np.uint8)
        result = main(img)