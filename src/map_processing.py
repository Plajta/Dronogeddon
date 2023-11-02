import time
import string

import cv2
import numpy as np
from sklearn.cluster import KMeans

LABELS = string.ascii_uppercase

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

drone_last_pos = (0, 0)
map_vis = np.full((1000, 1000, 3), 255, dtype='uint8')
map_data = np.zeros((1000, 1000))

xsize, ysize = map_data.shape

brum = [[378, -3], [378, -3], [378, -3], [380, -2], [380, 0], [381, 0], [383, 2], [384, 3], [393, 5], [393, 7], [393, 9], [396, 10], [399, 11], [403, 13], [392, 14], [296, 15], [281, 18], [276, 18], [276, 20], [275, 21], [272, 22], [438, 24], [444, 25], [448, 26], [458, 28], [461, 29], [476, 30], [478, 32], [482, 33], [497, 35], [497, 36], [477, 37], [477, 39], [458, 40], [455, 42], [452, 43], [451, 45], [416, 46], [416, 48], [405, 49], [399, 50], [399, 52], [434, 53], [442, 55], [442, 56], [468, 58], [486, 59], [481, 60], [481, 63], [526, 64], [719, 65], [719, 67], [807, 68], [1267, 69], [1257, 70], [884, 72], [872, 73], [854, 74], [465, 76], [465, 77], [462, 78], [461, 79], [461, 81], [461, 82], [461, 83], [461, 85], [463, 86], [463, 87], [463, 88], [466, 90], [467, 91], [471, 92], [471, 93], [471, 95], [463, 96], [460, 97], [399, 100], [383, 101], [377, 103], [378, 104], [469, 106], [441, 107], [441, 108], [434, 110], [391, 111], [382, 112], [350, 114], [350, 115], [348, 116], [344, 118], [341, 119], [340, 121], [322, 122], [271, 123], [271, 125], [263, 126], [253, 128], [251, 131], [251, 132], [242, 133], [239, 135], [237, 136], [229, 137], [226, 139], [218, 140], [214, 141], [214, 143], [212, 144], [205, 145], [204, 146], [203, 148], [203, 149], [198, 150], [197, 151], [197, 153], [197, 154], [196, 155], [190, 157], [190, 158], [188, 159], [187, 161], [187, 162], [186, 164], [186, 165], [186, 166], [184, 168], [184, 169], [184, 170], [184, 172], [184, 173], [184, 174], [185, 175], [185, 177], [186, 178], [187, 179], [187, -179], [187, -177], [187, -176], [188, -175], [190, -174], [191, -172], [193, -171], [193, -170], [197, -169], [197, -167], [198, -166], [201, -165], [205, -164], [205, -162], [207, -161], [211, -159], [215, -158], [219, -157], [219, -156], [221, -154], [221, -153], [223, -152], [233, -151], [238, -150], [239, -148], [241, -147], [249, -146], [253, -145], [244, -144], [244, -143], [240, -141], [238, -140], [237, -139], [229, -138], [222, -137], [220, -136], [219, -135], [208, -133], [208, -132], [206, -131], [202, -130], [200, -129], [200, -126], [200, -125], [199, -124], [198, -123], [191, -122], [189, -120], [187, -119], [187, -118], [183, -117], [183, -116], [180, -114], [179, -113], [179, -112], [179, -111], [176, -109], [174, -108], [174, -107], [174, -106], [174, -105], [171, -103], [171, -102], [171, -101], [171, -100], [171, -99], [171, -98], [171, -96], [171, -95], [171, -94], [172, -93], [172, -92], [173, -91], [174, -90], [174, -88], [175, -87], [175, -86], [176, -85], [178, -83], [178, -82], [178, -81], [179, -80], [183, -78], [184, -77], [185, -76], [187, -74], [187, -73], [192, -72], [195, -71], [195, -70], [196, -68], [197, -67], [198, -66], [201, -65], [201, -64], [214, -63], [214, -61], [215, -60], [216, -59], [218, -58], [226, -57], [227, -55], [250, -54], [250, -53], [350, -51], [351, -50], [353, -49], [353, -47], [339, -46], [327, -45], [322, -43], [321, -42], [321, -40], [320, -39], [368, -37], [368, -36], [391, -35], [396, -33], [403, -32], [411, -30], [406, -28], [405, -27], [402, -25], [392, -24], [391, -20], [387, -19], [384, -17], [383, -15], [383, -14], [381, -12], [376, -10], [374, -8], [374, -6], [374, -5], [374, -3]]

class MapProcessing:
    def __init__(self):
        self.tmp_map = np.zeros((1200, 1200), dtype=np.uint8)
        self.corners = []

    def map_init(self):
        global map_vis, map_data

        map_data = np.zeros((xsize, ysize))

        map_vis = np.full((xsize, ysize, 3), 255, dtype='uint8')

        map_vis = cv2.line(map_vis, (round(xsize/2-20), round(ysize/2-20)), (round(xsize/2+20), round(ysize/2+20)), (0, 255, 0), 2)
        map_vis = cv2.line(map_vis, (round(xsize/2+20), round(ysize/2-20)), (round(xsize/2-20), round(ysize/2+20)), (0, 255, 0), 2)
        # draw drone position (in center)
        pass


    def update_map(self, dist, curr_angle):
        global map_vis, map_data

        xpos = np.cos(-curr_angle/180*np.pi)*dist/2 + xsize/2
        ypos = np.sin(-curr_angle/180*np.pi)*dist/2 + ysize/2

        try:
            map_data[round(xpos), round(ypos)] = 1
            map_vis = cv2.circle(map_vis, (round(ypos), round(xpos)), 1, (0, 0, 0), 2)
        except IndexError as e:
            print(e)

        #cv2.imshow("Brum...", map_data)
        #cv2.imshow("Brumik.", map)

    def process_map(self, map_vis, map_d):
        line_d = np.zeros(map_d.shape[:2], dtype=np.uint8)

        #Dilation
        kernel = np.ones((8, 8), np.uint8) 
        map_d_dilated = cv2.dilate(map_d, kernel, iterations=1).astype(np.uint8) * 255
        
        #HoughLines
        linesP = cv2.HoughLinesP(map_d_dilated, 1, np.pi / 180, 50, None, 80, 40)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(map_vis, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                cv2.line(line_d, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)

        points = []

        #Get Extremes
        for line in linesP:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]

            points.append([x1, y1])
            points.append([x2, y2])

        #categorize extremes into clusters (kmeans)
        kmeans = KMeans(n_init="auto")
        kmeans.fit(points)

        labels = kmeans.labels_
        n_clusters = labels.max()

        point_clusters = [ [] for _ in range(n_clusters + 1) ]
        representative_points = []

        for i, point in enumerate(points):
            label = labels[i]

            point_clusters[label].append(point)

            #cv2.putText(map_vis, LABELS[label], (point[0], point[1]), font, fontScale, color, thickness, cv2.LINE_AA) 

        #take median from all cluster values
        for point_cluster in point_clusters:
            cluster_np = np.array(point_cluster)

            x_values = cluster_np[:, 0]
            y_values = cluster_np[:, 1]

            x_median = round(np.median(x_values))
            y_median = round(np.median(y_values))

            representative_points.append([x_median, y_median])

        for point in representative_points:
            cv2.circle(map_vis, point, 8, (255, 0, 0), -1)

        cv2.imshow("map", map_vis)
        cv2.imshow("map_dilated", map_d_dilated)
        cv2.imshow("line_data", line_d)

if __name__ == "__main__":
    proc_instance = MapProcessing()

    proc_instance.map_init()
    for i in brum:
        proc_instance.update_map(*i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    proc_instance.process_map(map_vis, map_data)
    
    while not cv2.waitKey(0) & 0xFF == ord('q'):
        pass

