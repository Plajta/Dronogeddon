import cv2
import numpy as np
import math

def K_Clustering(image, n_clusters):
    """
    IN: BGR image, K (number of clusters)
    OUT: BGR image
    """

    Reshape = image.reshape((-1,3))
    Reshape = np.float32(Reshape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Reshape, n_clusters, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_res = res.reshape((image.shape))

    return img_res

def HoughLines(image_in, image_out):
    """
    IN: binary image
    OUT: BGR image
    """

    lines = cv2.HoughLines(image_in, 1, np.pi / 180, 135, None, 0, 0)
    #must be binary (0 or 1)

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
            cv2.line(image_out, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return image_out

def Adjust_Gamma(image, gamma):
    """
    IN: BGR image, gamma
    OUT: BGR image with different Gamma
    """

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def Automatic_Thresh(gray_image): #https://stackoverflow.com/questions/41893029/opencv-canny-edge-detection-not-working-properly
    """
    IN: grayscale image
    OUT: lower, upper threshold (for edge detection and thresholding)
    """

    v = np.median(gray_image)
    sigma = 0.33

    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    return lower, upper

def Detect_Corners(image_in, image_out):
    """
    IN: grayscale image
    OUT: BGR image
    """

    gray_image = np.float32(image_in)
    corners = cv2.cornerHarris(image_in, 2, 5, 0.07)
  
    # Results are marked through the dilated corners
    corners = cv2.dilate(corners, None)

    image_out[corners > 0.01 * corners.max()]=[0, 0, 255]
    return image_out

#main code

vid = cv2.VideoCapture(0)
while(True):
      
    ret, frame = vid.read()

    #creating blank images
    blank = np.zeros(frame.shape)

    #clustering
    img = Adjust_Gamma(frame, 0.4)
    img_cluster = K_Clustering(frame, 5) #TODO: maybe

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #edge detection
    upper, lower = Automatic_Thresh(img_gray)
    img_canny = cv2.Canny(img_gray, lower, upper)
    
    blank = HoughLines(img_canny, blank)

    #corner detection
    blank = Detect_Corners(img_gray, blank)

    cv2.imshow("frame_canny", img_canny)
    cv2.imshow("frame_lines", blank)
    cv2.imshow("frame_K_means", img_cluster)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break