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

def HoughLines(image_in, image_out, l_color):
    """
    IN: binary image
    OUT: BGR image
    """

    lines = cv2.HoughLines(image_in, 1, np.pi / 180, 135, None, 0, 0)
    line_points = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) #x, y notation
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image_out, pt1, pt2, l_color, 3, cv2.LINE_AA)

            line_points.append([pt1, pt2])

    return image_out, line_points

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
    OUT: binary image
    """

    image_in = np.float32(image_in)
    corners = cv2.cornerHarris(image_in, 2, 5, 0.07)
  
    # Results are marked through the dilated corners
    corners = cv2.dilate(corners, None)

    image_out[corners > 0.01 * corners.max()] = 255
    return image_out

def Slice_image(image_in, strides):
    """
    IN: binary image, stride (y, x)
    OUT: sliced binary image
    """

    sliceY = math.ceil(image_in.shape[0] / strides[0])
    sliceX = math.ceil(image_in.shape[1] / strides[1])

    for iy in range(strides[0]): cv2.line(image_in, (iy * sliceY, 0), (iy * sliceY, image_in.shape[0]), 0, 3)
    for ix in range(strides[1]): cv2.line(image_in, (0, ix * sliceX), (image_in.shape[1], ix * sliceX), 0, 3)

    return image_in

def CalculateAngle(line):
    d_x = line[0][0] - line[1][0]
    d_y = line[0][1] - line[1][1]

    deg = round(math.degrees(math.atan(abs(d_x) / abs(d_y))), 2)

    if d_x < 0 or d_y < 0:
        #is negative and so the actual angle is also negative
        line.append(-deg)
    else: line.append(deg)

def GroupLines(lines):
    pass #TODO: dodělat!

#main code

vid = cv2.VideoCapture(0)
while(True):
      
    ret, frame = vid.read()

    #creating blank images and other image manipulation
    blank = np.zeros(frame.shape)
    corners = np.zeros(frame.shape[:2], dtype=np.uint8)

    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    #
    # Generating Canny, Harris and K-Means
    #

    #clustering
    img = Adjust_Gamma(frame, 0.4)
    img_cluster = K_Clustering(frame, 7) #TODO: maybe

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #edge detection
    upper, lower = Automatic_Thresh(img_gray)
    img_canny = cv2.Canny(img_gray, lower, upper)

    img_canny = Slice_image(img_canny, (5, 5))

    #corner detection
    corners = Detect_Corners(img_gray, corners)

    #
    # Fitting lines through interest points
    #

    blank, lines1 = HoughLines(img_canny, blank, (0, 0, 255))
    blank, lines2 = HoughLines(corners, blank, (255, 0, 0)) #TODO: kinda useless

    #
    # Line grouping by difference of angle and distance
    #
    
    lines = lines1 + lines2
    for line in lines:
        CalculateAngle(line)


    
    print(lines)

    #cv2.imshow("frame", frame)
    cv2.imshow("frame_canny", img_canny)
    #cv2.imshow("frame_corners", corners)
    cv2.imshow("frame_lines", blank)
    cv2.imshow("frame_K_means", img_cluster)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break