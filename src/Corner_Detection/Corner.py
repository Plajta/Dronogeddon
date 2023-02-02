import cv2
import numpy as np
import math

def K_Clustering(image, n_clusters):
    Reshape = image.reshape((-1,3))
    Reshape = np.float32(Reshape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Reshape, n_clusters, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_res = res.reshape((image.shape))

    return img_res

def Adjust_Gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

vid = cv2.VideoCapture(0)
  
while(True):
      
    ret, frame = vid.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Adjust_Gamma(frame, 1.0)

    img_cluster = K_Clustering(img, 5) #TODO: maybe
    img_canny = cv2.Canny(img_gray, 0, 255)
    
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


    lines = cv2.HoughLines(img_canny, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', img)
    cv2.imshow("frame_canny", img_sobel)
    #cv2.imshow("frame_lines", frame)
    #cv2.imshow("frame_K_means", img_cluster)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break