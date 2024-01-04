import time
import string
import math

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KDTree
from sklearn.metrics import silhouette_score

LABELS = string.ascii_uppercase

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

MAP_WIDTH = 1000
drone_last_pos = (round(MAP_WIDTH / 2), round(MAP_WIDTH / 2)) #TODO pak změň - teď je to default

map_vis = np.full((MAP_WIDTH, MAP_WIDTH, 3), 255, dtype='uint8')
map_data = np.zeros((MAP_WIDTH, MAP_WIDTH))

xsize, ysize = map_data.shape

brum = [[[182, 317, 270, 257, 3730], -107], [[182, 317, 271, 257, 3716], -107], [[182, 316, 271, 257, 3724], -107], [[182, 316, 271, 256, 3721], -106], [[182, 316, 272, 256, 3740], -104], [[183, 316, 273, 256, 3511], -103], [[184, 317, 273, 256, 3578], -101], [[242, 316, 275, 256, 2122], -99], [[242, 316, 275, 256, 2122], -97], [[308, 318, 278, 258, 1811], -96], [[309, 321, 283, 262, 1790], -94], [[319, 324, 286, 265, 1654], -92], [[319, 324, 286, 265, 1654], -91], [[331, 329, 292, 271, 1245], -87], [[331, 329, 292, 271, 1245], -85], [[331, 339, 299, 275, 3253], -84], [[360, 347, 301, 277, 1608], -82], [[369, 352, 308, 278, 2258], -80], [[395, 356, 312, 275, 1622], -78], [[395, 356, 312, 275, 1622], -77], [[394, 371, 324, 218, 1835], -75], [[390, 374, 330, 187, 2612], -73], [[390, 374, 330, 187, 2612], -72], [[399, 385, 336, 181, 3466], -70], [[389, 386, 343, 179, 2555], -68], [[390, 387, 347, 180, 2556], -66], [[343, 388, 345, 181, 432], -65], [[334, 384, 343, 182, 173], -63], [[334, 384, 343, 182, 173], -61], [[334, 375, 339, 182, 128], -59], [[65535, 360, 337, 182, 103], -58], [[344, 345, 336, 181, 477], -56], [[657, 339, 332, 180, 344], -54], [[657, 339, 332, 180, 344], -52], [[321, 326, 316, 176, 1263], -50], [[321, 326, 316, 176, 1263], -49], [[314, 320, 316, 176, 14425], -47], [[327, 316, 317, 176, 756], -45], [[307, 313, 319, 183, 1658], -44], [[307, 313, 319, 183, 1658], -42], [[282, 307, 318, 231, 6655], -40], [[279, 300, 320, 315, 6849], -39], [[279, 300, 320, 318, 6849], -37], [[278, 295, 319, 318, 6911], -35], [[274, 295, 319, 318, 7200], -33], [[272, 292, 318, 318, 7311], -32], [[272, 292, 318, 318, 7311], -30], [[271, 289, 315, 317, 7624], -28], [[270, 287, 314, 317, 8230], -26], [[270, 282, 301, 316, 8426], -25], [[269, 257, 276, 315, 8844], -23], [[269, 201, 267, 314, 8841], -21], [[269, 193, 261, 313, 5815], -19], [[269, 193, 261, 313, 5815], -18], [[270, 187, 261, 315, 5706], -16], [[271, 186, 264, 315, 4082], -14], [[274, 184, 262, 317, 3488], -13], [[274, 184, 262, 317, 3488], -11], [[275, 185, 261, 320, 3715], -10], [[276, 187, 261, 321, 3373], -8], [[280, 189, 262, 324, 3981], -6], [[312, 192, 263, 324, 8213], -5], [[283, 198, 265, 328, 3087], -3], [[285, 209, 267, 331, 2773], -1], [[290, 278, 272, 333, 3029], 0], [[304, 299, 275, 334, 3112], 1], [[304, 299, 275, 334, 3112], 4], [[305, 313, 277, 342, 3066], 6], [[317, 319, 279, 347, 3149], 8], [[327, 323, 281, 352, 3717], 9], [[335, 330, 284, 358, 4051], 11], [[343, 342, 281, 363, 3764], 13], [[352, 351, 272, 364, 2132], 14], [[352, 351, 272, 364, 2132], 16], [[345, 366, 206, 373, 2655], 18], [[349, 371, 189, 377, 2802], 19], [[365, 375, 187, 376, 3125], 21], [[349, 375, 187, 376, 3528], 23], [[346, 373, 187, 383, 3593], 24], [[345, 374, 187, 375, 3607], 26], [[315, 373, 188, 375, 2196], 27], [[311, 365, 189, 350, 2267], 29], [[314, 354, 190, 341, 2324], 31], [[314, 354, 190, 341, 2324], 32], [[313, 345, 189, 334, 2625], 34], [[323, 321, 189, 330, 2222], 36], [[323, 321, 189, 330, 2222], 37], [[324, 315, 188, 325, 2251], 39], [[325, 308, 186, 320, 2269], 40], [[332, 307, 185, 316, 1025], 42], [[336, 298, 186, 310, 1264], 44], [[327, 287, 225, 301, 4861], 45], [[327, 287, 225, 301, 4861], 47], [[329, 283, 307, 297, 2437], 48], [[329, 283, 307, 297, 2437], 52], [[349, 277, 326, 291, 618], 53], [[381, 273, 326, 291, 166], 55], [[65535, 268, 327, 283, 80], 60], [[65535, 266, 328, 280, 53], 61], [[349, 266, 327, 264, 5277], 63], [[346, 264, 325, 236, 2118], 65], [[330, 263, 326, 188, 1972], 66], [[330, 263, 326, 188, 1972], 68], [[270, 263, 325, 182, 3583], 69], [[263, 262, 324, 177, 7810], 71], [[263, 263, 325, 175, 7775], 73], [[264, 262, 325, 174, 7760], 74], [[264, 263, 325, 173, 7759], 76], [[270, 263, 325, 173, 7287], 77], [[274, 264, 328, 175, 4861], 79], [[274, 264, 328, 175, 4861], 81], [[274, 268, 330, 179, 4412], 82], [[278, 269, 332, 183, 5026], 84], [[285, 271, 337, 189, 5315], 86], [[285, 271, 337, 189, 5315], 87], [[288, 277, 337, 212, 5989], 89], [[288, 278, 342, 292, 6012], 91], [[302, 281, 347, 298, 5218], 93], [[302, 281, 347, 308, 5218], 94], [[304, 289, 357, 308, 5158], 96], [[304, 292, 353, 314, 5129], 98], [[434, 298, 363, 320, 200], 100], [[462, 303, 372, 334, 247], 101], [[350, 316, 374, 350, 2469], 103], [[216, 325, 375, 357, 23032], 105], [[188, 338, 384, 359, 8830], 107], [[190, 342, 383, 364, 8513], 108], [[195, 344, 383, 366, 7781], 110], [[198, 343, 377, 362, 7481], 112], [[205, 345, 378, 361, 7905], 114], [[205, 345, 378, 361, 7905], 115], [[205, 346, 359, 361, 7725], 117], [[199, 346, 345, 351, 7386], 119], [[193, 344, 344, 342, 7798], 121], [[189, 337, 337, 318, 8182], 123], [[183, 334, 337, 301, 8635], 124], [[184, 329, 331, 296, 7578], 126], [[186, 331, 329, 293, 4598], 128], [[812, 330, 321, 290, 870], 130], [[812, 330, 321, 290, 870], 131], [[806, 327, 316, 282, 823], 133], [[511, 324, 309, 280, 866], 135], [[337, 324, 307, 277, 4775], 137], [[337, 320, 307, 274, 4784], 138], [[331, 323, 304, 271, 5007], 140], [[331, 323, 304, 271, 5007], 142], [[328, 318, 298, 268, 5114], 144], [[325, 317, 297, 265, 5251], 145], [[324, 332, 293, 265, 5308], 147], [[324, 332, 293, 265, 5308], 149], [[323, 333, 289, 264, 5610], 151], [[321, 335, 280, 263, 5799], 152], [[321, 333, 272, 263, 5793], 154], [[321, 329, 212, 262, 5786], 156], [[321, 329, 203, 262, 5248], 157], [[322, 328, 197, 262, 4870], 158], [[322, 329, 191, 262, 3737], 161], [[324, 325, 190, 262, 3796], 162], [[325, 325, 188, 262, 3451], 162], [[325, 325, 188, 262, 3451], 163], [[326, 326, 186, 263, 2051], 166], [[327, 328, 185, 263, 2276], 167], [[327, 290, 185, 263, 2758], 168], [[329, 288, 186, 265, 1859], 169], [[331, 269, 187, 266, 1820], 170], [[334, 260, 187, 267, 2797], 171], [[336, 260, 190, 268, 3228], 172], [[337, 259, 192, 270, 3212], 174], [[339, 262, 195, 270, 2464], 175], [[344, 263, 200, 273, 2073], 176], [[346, 265, 207, 275, 2822], 177], [[349, 265, 216, 275, 2720], 178], [[356, 267, 289, 277, 2326], 179], [[356, 267, 289, 277, 2326], -179], [[358, 270, 304, 281, 2491], -178], [[360, 272, 309, 282, 2510], -177], [[365, 275, 314, 285, 2219], -175], [[375, 275, 316, 287, 3019], -174], [[377, 279, 320, 292, 3284], -173], [[394, 281, 322, 296, 3039], -171], [[394, 281, 322, 296, 3039], -170], [[403, 285, 339, 305, 2824], -169], [[405, 288, 353, 309, 2772], -167], [[415, 285, 362, 315, 2625], -166], [[428, 270, 366, 323, 2650], -164], [[429, 203, 369, 326, 2684], -162], [[412, 190, 372, 331, 560], -161], [[372, 188, 373, 335, 3184], -156], [[372, 188, 373, 335, 3184], -154], [[371, 189, 365, 335, 3364], -152], [[363, 189, 360, 327, 1946], -150], [[353, 189, 347, 324, 2127], -148], [[337, 188, 320, 321, 2438], -147], [[334, 188, 315, 316, 2580], -145], [[324, 188, 300, 313, 2233], -143], [[324, 187, 298, 303, 2081], -141], [[319, 186, 297, 307, 2900], -139], [[319, 186, 297, 307, 2900], -137], [[313, 185, 290, 307, 2927], -136], [[304, 186, 286, 307, 3329], -134], [[303, 223, 280, 311, 3674], -132], [[302, 314, 278, 312, 3624], -130], [[294, 322, 278, 310, 2561], -128], [[294, 322, 278, 310, 2561], -124], [[292, 326, 273, 312, 3138], -123], [[292, 323, 271, 330, 3702], -119], [[292, 323, 271, 330, 3702], -117], [[183, 322, 269, 329, 10675], -115], [[183, 322, 269, 329, 10675], -113], [[184, 319, 268, 323, 9184], -111], [[183, 319, 268, 321, 11049], -109], [[184, 319, 268, 319, 10295], -107]]
class MapProcessing:
    def __init__(self):
        self.tmp_map = np.zeros((1200, 1200), dtype=np.uint8)
        self.corners = []

    def map_init(self):
        global map_vis, map_data

        map_data = np.zeros((xsize, ysize))

        map_vis = np.full((xsize, ysize, 3), 255, dtype='uint8')

        # draw drone position (in center)
        map_vis = cv2.line(map_vis, (round(xsize/2-20), round(ysize/2-20)), (round(xsize/2+20), round(ysize/2+20)), (0, 255, 0), 2)
        map_vis = cv2.line(map_vis, (round(xsize/2+20), round(ysize/2-20)), (round(xsize/2-20), round(ysize/2+20)), (0, 255, 0), 2)

        # draw scale
        length = 80
        xstart = 20
        ystart = 50
        color = (63, 63, 63)
        map_vis = cv2.putText(map_vis, f"{length} cm", (xstart, ystart-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color)
        map_vis = cv2.line(map_vis, (xstart, ystart), (xstart+length, ystart), color, 1)
        map_vis = cv2.line(map_vis, (xstart, ystart-10), (xstart, ystart+10), color, 1)
        map_vis = cv2.line(map_vis, (xstart+length, ystart-10), (xstart+length, ystart+10), color, 1)
        pass

    def update_map(self, dist, curr_angle, color):
        global map_vis, map_data

        xpos = np.cos(-curr_angle/180*np.pi)*dist/2 + xsize/2
        ypos = np.sin(-curr_angle/180*np.pi)*dist/2 + ysize/2

        try:
            map_data[round(xpos), round(ypos)] = 1
            map_vis = cv2.circle(map_vis, (round(ypos), round(xpos)), 1, color, 2)
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
        linesP = cv2.HoughLinesP(map_d_dilated, 1, np.pi / 180, 50, None, 25, 50)
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

        return points, line_d, linesP

    def get_waypoints_kmeans_triangles(self, points):
        #categorize extremes into clusters (kmeans) and predict number of points using silhouette method
        lim = round(len(points) / 8)

        scores = []
        for k in range(2, lim + 1):
            kmeans = KMeans(n_init="auto", init="k-means++", n_clusters=16)
            kmeans.fit(points)
            pred = kmeans.predict(points)
            score = silhouette_score(points, pred)
            scores.append(score)

        kmeans = KMeans(n_init="auto", init="k-means++", n_clusters=lim)
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

        return representative_points

    def cluster_DBSCAN(self, points):
        #categorize extremes into clusters (kmeans) and predict number of points using silhouette method
        clustering = DBSCAN(eps=30, min_samples=5).fit(points)

        labels = clustering.labels_
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

        cv2.imshow("DBSCAN", map_vis)
        cv2.waitKey(0)

        return representative_points

    def get_waypoints_by_decreasing_triangles(self, representative_points):

        #transform representative points to dict
        iterated_points = []

        #process representative points to get triangles
        centers = []

        for i, point in enumerate(representative_points):
            if i == len(representative_points) - 2:
                break

            distances = []
            points_copy = representative_points.copy()
            points_copy.pop(i)

            for search_point in points_copy:
                if search_point in iterated_points:
                    continue

                d_x = abs(point[0] - search_point[0])
                d_y = abs(point[1] - search_point[1])

                distances.append([round(math.sqrt(d_x ** 2 + d_y ** 2), 2), search_point[0], search_point[1]])

            #sort distances 
            distances_np = np.array(distances)
            distances_sorted = distances_np[distances_np[:, 1].argsort()]

            #get two triangle points
            a1 = distances_sorted[0][1:].astype(np.uint64)
            a2 = distances_sorted[1][1:].astype(np.uint64)

            centers.append([round((point[0]+a1[0]+a2[0])/3), round((point[1]+a1[1]+a2[1])/3)])

            iterated_points.append(point)

            cv2.circle(map_vis, centers[i], 8, (0, 0, 0), -1)
            cv2.circle(map_vis, point, 8, (255, 255, 0), -1)

            cv2.line(map_vis, point, a1, (0, 255, 0), 3) 
            cv2.line(map_vis, a1, a2, (0, 255, 0), 3) 
            cv2.line(map_vis, a2, point, (0, 255, 0), 3)

            cv2.imshow("test", map_vis)
            cv2.waitKey(0)

    def get_waypoints_by_triangles(self, representative_points, line_map):
        centers = []
        MINIMAL_WALL_DISTANCE = 20
        
        for i, point in enumerate(representative_points):
            if i == len(representative_points) - 2:
                break

            distances = []
            points_copy = representative_points.copy()
            for i2 in range(i):
                points_copy.pop(0)

            for search_point in points_copy:
                d_x = abs(point[0] - search_point[0])
                d_y = abs(point[1] - search_point[1])

                distances.append([round(math.sqrt(d_x ** 2 + d_y ** 2), 2), search_point[0], search_point[1]])
                
            #sort distances 
            distances_np = np.array(distances)
            distances_sorted = distances_np[distances_np[:, 1].argsort()]

            #get two triangle points
            for i, dist1 in enumerate(distances_sorted):
                distances_sorted2 = np.delete(distances_sorted, (i), axis=0)
                for dist2 in distances_sorted2:
                
                    a1 = dist1[1:].astype(np.uint64)
                    a2 = dist2[1:].astype(np.uint64)

                    center = [round((point[0]+a1[0]+a2[0])/3), round((point[1]+a1[1]+a2[1])/3)]
                    centers.append(center)

                    cv2.circle(map_vis, point, 8, (255, 255, 0), -1)

                    cv2.line(map_vis, point, a1, (0, 255, 0), 3) 
                    cv2.line(map_vis, a1, a2, (0, 255, 0), 3) 
                    cv2.line(map_vis, a2, point, (0, 255, 0), 3)

        #get rid of points overlapping points
        true_centers = []
        for center in centers:
            if line_map[center[1], center[0]] != 255:
                true_centers.append(center)

        centers = []
        for center in true_centers:
            n_white_pix = np.sum(line_data[center[1]-MINIMAL_WALL_DISTANCE:center[1]+MINIMAL_WALL_DISTANCE, center[0]-MINIMAL_WALL_DISTANCE:center[0]+MINIMAL_WALL_DISTANCE] == 255)
            if n_white_pix <= 0:
                centers.append(center)

        for center in centers:
            cv2.circle(map_vis, center, 8, (0, 0, 0), -1)

        return centers

    def get_room_openings(self, points, line_data):
        sample_size = 50
        outer_width = 50
        inner_width = 35

        room_opening_points = []

        for point in points:
            #start to iterate over pixels in circle-like structure
            line_sample = line_data[point[1] - sample_size:point[1] + sample_size, point[0] - sample_size:point[0] + sample_size]
            point_sample = np.zeros(line_sample.shape, dtype=np.uint8)
            cv2.circle(point_sample, (sample_size, sample_size), outer_width, (255, 255, 255), -1) #create outer circle for masking
            cv2.circle(point_sample, (sample_size, sample_size), inner_width, (0, 0, 0), -1) #reate inner circle to later get different objects

            #mask these so you can get openings
            masked = cv2.bitwise_and(line_sample, point_sample)

            #get contours so you can get number of walls
            contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            n_walls = len(contours)
            if n_walls == 1:
                #found opening!
                room_opening_points.append(point)

        return room_opening_points
    
    def find_openings_using_lowest_distance(self, points):
        distances = []
        for i, point in enumerate(points):
            points_copy = points.copy()
            points_copy.pop(i)

            for point_copy in points_copy:
                c = round(math.sqrt(abs(point[0] - point_copy[0])**2 + abs(point[1] - point_copy[1])**2), 2)
                distances.append([c, point, point_copy])

        min_num = 0
        curr_i = 0
        for i, dist in enumerate(distances):
            if i == 0:
                min_num = dist[0]

            if dist[0] <= min_num:
                curr_i = i

        return distances[curr_i][1:]

class Graph:

    def __check_segment__(self, p, q, r):
        pass

    def __check_orientation__(self, p, q, r):
        pass

    def __check_for_colisions__(self, line_1, line_2):
        o1 = self.__check_orientation__(line_1[0], line_1[1], line_2[0]) 
        o2 = self.__check_orientation__(line_1[0], line_1[1], line_2[1]) 
        o3 = self.__check_orientation__(line_2[0], line_2[1], line_1[0]) 
        o4 = self.__check_orientation__(line_2[0], line_2[1], line_1[1]) 
    
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # Special Cases 
    
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
        if ((o1 == 0) and self.__check_segment__(line_1[0], line_2[0], line_1[1])): 
            return True
    
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
        if ((o2 == 0) and self.__check_segment__(line_1[0], line_2[1], line_1[1])): 
            return True
    
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
        if ((o3 == 0) and self.__check_segment__(line_2[0], line_1[0], line_2[1])): 
            return True
    
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
        if ((o4 == 0) and self.__check_segment__(line_2[0], line_1[1], line_2[1])): 
            return True
    
        # If none of the cases 
        return False

    def __init__(self, points, dest_points, drone_pos, lines):
        self.VERTICAL_PASS_COEF = 2
        self.MIN_POINT_DIST = 30

        self.path_array = []

        self.start_point = drone_pos
        self.end_point = [round((dest_points[0][0] + dest_points[1][0])/2), round((dest_points[0][1] + dest_points[1][1])/2)]
        
        points_copy = points.copy()
        while True:
            if len(points_copy) == 0:
                break

            point = points_copy[0]
            points_copy.pop(0)
            for point2 in points:
                
                #iterate on every line to check colision
                for line in lines:
                    point_path = [point, point2]
                    line_path = [line[0][:2].tolist(), line[0][2:].tolist()]
                    self.__check_for_colisions__(point_path, line_path)

                    exit(0)



class Astar:
    def __init__(self):
        pass

    def process_points(self, graph):
        pass

if __name__ == "__main__":
    proc_instance = MapProcessing()
    algorithm = Astar()
    

    proc_instance.map_init()
    for (f, l, r, b, qual), deg in brum:
        proc_instance.update_map(f, deg, (0, 0, 0))
        proc_instance.update_map(b, deg+180, (0, 128, 128))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    point_data, line_data, lines = proc_instance.process_map(map_vis, map_data)
    clustered_points = proc_instance.cluster_DBSCAN(point_data)
    opening_points = proc_instance.get_room_openings(clustered_points, line_data)
    opening = proc_instance.find_openings_using_lowest_distance(opening_points)
    
    waypoints = proc_instance.get_waypoints_by_triangles(clustered_points, line_data)

    #now to path construction
    directed_graph = Graph(waypoints, opening, drone_last_pos, lines)
    algorithm.process_points(directed_graph)

    #proc_instance.construct_path(opening, clustered_points)

    for point in opening:
        cv2.circle(map_vis, point, 8, (125, 125, 0), -1)


    cv2.imshow("test", map_vis)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()        

