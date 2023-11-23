import cv2
import numpy as np

img_path = "/media/work/Workspace/Projects/Dronogeddon/src/neural/dataset/test/images/Door0224_png.rf.59c93311feb17ae4a5d5c677f405d4c8.jpg"

image = cv2.imread(img_path)
contours_draw = np.zeros(image.shape)

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#img_gray = cv2.GaussianBlur(img_gray, (7,7), 0)

#thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY) 
edges = cv2.Canny(thresh, 100, 200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_areas = []
accepted_contours = []
for contour_object in contours:
    contour_areas.append(cv2.contourArea(contour_object))

median = np.median(np.array(contour_areas))
for i, contour_area in enumerate(contour_areas):

    if contour_area > median:
        accepted_contours.append(contours[i])

cv2.drawContours(contours_draw, accepted_contours, -1, (0,255,0), 3)

cv2.imshow("test", edges)
cv2.imshow("test-draw", contours_draw)
cv2.waitKey(0)