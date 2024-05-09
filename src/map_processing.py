import math
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# visualisation
import matplotlib.pyplot as plt
import networkx as nx

from CustomAStar import CustomAStar

# font 
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale 
fontScale = 1

# Blue color in BGR 
color = (255, 0, 0)

# Line thickness of 2 px 
thickness = 2

# for testing
rho = 1
theta_add = 180
thresh = 50
lines = None
minLineLength = 25
maxLineGap = 50


class Drone:
    def __init__(self, drone_angle, drone_position):
        self.drone_angle = drone_angle
        self.drone_position = drone_position


class MapProcessing:
    def __init__(self, inp_data, map_shape, drone_pos, drone_angle, debug):
        self.tmp_map = np.zeros((map_shape), dtype=np.uint8)
        self.corners = []
        self.input_data = inp_data
        self.map_width = map_shape[0]
        self.map_height = map_shape[1]
        self.drone_heading = 0  # in degrees
        self.debug_mode = debug
        self.drone_pos = drone_pos
        self.drone_angle = drone_angle

        self.map_vis = np.full((self.map_width, self.map_height, 3), 255, dtype='uint8')
        self.map_data = np.zeros((self.map_width, self.map_height))

        self.map_init()
        for (f, l, r, b, qual, qual2), deg in self.input_data:
            coords_f = self.update_map(f, deg)
            coords_b = self.update_map(b, deg + 180)

        data_dict: {int: {str: int}} = {}
        deg_offsets = {'f': 360, 'l': 270, 'r': 450, 'b': 180}

        for (f, l, r, b, qual, qual2), deg in self.input_data:
            dists = {'f': f, 'l': l, 'r': r, 'b': b}
            for key, offset in deg_offsets.items():
                data_dict.setdefault((deg + offset) % 360, {})
                value = data_dict[(deg + offset) % 360].get(key)
                data_dict[(deg + offset) % 360][key] = dists[key] if value is None else (value + dists[key]) / 2

        for deg, dists in data_dict.items():
            dist = sum(dists.values()) / len(dists)
            merged_coords = self.update_map(dist, deg)
            self.write_to_vis(merged_coords, (0, 0, 0))

        self.tmp_map = self.map_vis.copy()

        if self.debug_mode:
            cv2.imshow("test", self.map_vis)
            cv2.waitKey(0)

    def map_init(self):

        # draw drone position (in center)
        self.map_vis = cv2.line(self.map_vis, (self.drone_pos[0] - 20, self.drone_pos[1] - 20),
                                (self.drone_pos[0] + 20, self.drone_pos[1] + 20), (0, 255, 0), 2)
        self.map_vis = cv2.line(self.map_vis, (self.drone_pos[0] + 20, self.drone_pos[1] - 20),
                                (self.drone_pos[0] - 20, self.drone_pos[1] + 20), (0, 255, 0), 2)

        # draw scale
        length = 80
        xstart = 20
        ystart = 50
        color = (63, 63, 63)
        self.map_vis = cv2.putText(self.map_vis, f"{length} cm", (xstart, ystart - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                   color)
        self.map_vis = cv2.line(self.map_vis, (xstart, ystart), (xstart + length, ystart), color, 1)
        self.map_vis = cv2.line(self.map_vis, (xstart, ystart - 10), (xstart, ystart + 10), color, 1)
        self.map_vis = cv2.line(self.map_vis, (xstart + length, ystart - 10), (xstart + length, ystart + 10), color, 1)

    def update_map(self, dist, curr_angle):
        xpos = np.cos(-curr_angle / 180 * np.pi) * dist / 2 + self.map_width / 2
        ypos = np.sin(-curr_angle / 180 * np.pi) * dist / 2 + self.map_height / 2

        return [xpos, ypos]

    def write_to_vis(self, coords, color):
        try:
            self.map_data[round(coords[0]), round(coords[1])] = 1
            self.map_vis = cv2.circle(self.map_vis, (round(coords[0]), round(coords[1])), 1, color, 2)
        except IndexError as e:
            print(e)

    def process_map(self, map_d, hough_args=[]):
        # Hough args are used for testing, they consist of:
        # rho -> float
        # theta -> float
        # threshold -> int
        # lines -> None
        # minLineLength -> float
        # maxLineGap -> float

        line_d = np.zeros(map_d.shape[:2], dtype=np.uint8)

        # Dilation
        kernel = np.ones((8, 8), np.uint8)
        map_d_dilated = cv2.dilate(map_d, kernel, iterations=1).astype(np.uint8) * 255

        # HoughLines
        if len(hough_args) != 0:
            linesParametric = cv2.HoughLinesP(map_d_dilated, *hough_args)
        else:
            linesParametric = cv2.HoughLinesP(map_d_dilated, 1, np.pi / 180, 50, None, 25, 50)

        if linesParametric is None:
            print("No borders found :(")
            exit(0)

        # what (TODO)
        linesP = []
        for i in range(len(linesParametric)):
            line = linesParametric[i][0]
            linesP.append([line[1], line[0], line[3], line[2]])

        for i in range(0, len(linesP)):
            l = linesP[i]
            cv2.line(self.map_vis, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(line_d, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)

        points = []

        # Get Extremes
        for line in linesP:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]

            points.append([x1, y1])
            points.append([x2, y2])

        return points, line_d, linesP

    def get_waypoints_kmeans_triangles(self, points):
        # categorize extremes into clusters (kmeans) and predict number of points using silhouette method
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

        point_clusters = [[] for _ in range(n_clusters + 1)]
        representative_points = []

        for i, point in enumerate(points):
            label = labels[i]

            point_clusters[label].append(point)

        # take median from all cluster values
        for point_cluster in point_clusters:
            cluster_np = np.array(point_cluster)

            x_values = cluster_np[:, 0]
            y_values = cluster_np[:, 1]

            x_median = round(np.median(x_values))
            y_median = round(np.median(y_values))

            representative_points.append([x_median, y_median])

        for point in representative_points:
            cv2.circle(self.map_vis, point, 8, (255, 0, 0), -1)

        return representative_points

    def cluster_DBSCAN(self, points):
        # categorize extremes into clusters (kmeans) and predict number of points using silhouette method
        clustering = DBSCAN(eps=30, min_samples=5).fit(points)

        labels = clustering.labels_
        n_clusters = labels.max()

        point_clusters = [[] for _ in range(n_clusters + 1)]
        representative_points = []

        for i, point in enumerate(points):
            label = labels[i]

            point_clusters[label].append(point)

        # take median from all cluster values
        for point_cluster in point_clusters:
            cluster_np = np.array(point_cluster)

            x_values = cluster_np[:, 0]
            y_values = cluster_np[:, 1]

            x_median = round(np.median(x_values))
            y_median = round(np.median(y_values))

            representative_points.append([x_median, y_median])

        for point in representative_points:
            cv2.circle(self.map_vis, point, 8, (255, 0, 0), -1)

        cv2.imshow("DBSCAN", self.map_vis)
        cv2.waitKey(0)

        return representative_points

    def get_waypoints_by_decreasing_triangles(self, representative_points):

        # transform representative points to dict
        iterated_points = []

        # process representative points to get triangles
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

            # sort distances
            distances_np = np.array(distances)
            distances_sorted = distances_np[distances_np[:, 1].argsort()]

            # get two triangle points
            a1 = distances_sorted[0][1:].astype(np.uint64)
            a2 = distances_sorted[1][1:].astype(np.uint64)

            centers.append([round((point[0] + a1[0] + a2[0]) / 3), round((point[1] + a1[1] + a2[1]) / 3)])

            iterated_points.append(point)

            cv2.circle(self.map_vis, centers[i], 8, (0, 0, 0), -1)
            cv2.circle(self.map_vis, point, 8, (255, 255, 0), -1)

            cv2.line(self.map_vis, point, a1, (0, 255, 0), 3)
            cv2.line(self.map_vis, a1, a2, (0, 255, 0), 3)
            cv2.line(self.map_vis, a2, point, (0, 255, 0), 3)

            cv2.imshow("test", self.map_vis)
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

            # sort distances
            distances_np = np.array(distances)
            distances_sorted = distances_np[distances_np[:, 1].argsort()]

            # get two triangle points
            for i, dist1 in enumerate(distances_sorted):
                distances_sorted2 = np.delete(distances_sorted, (i), axis=0)
                for dist2 in distances_sorted2:
                    a1 = dist1[1:].astype(np.uint64)
                    a2 = dist2[1:].astype(np.uint64)

                    center = [round((point[0] + a1[0] + a2[0]) / 3), round((point[1] + a1[1] + a2[1]) / 3)]
                    centers.append(center)

                    cv2.circle(self.map_vis, point, 8, (255, 255, 0), -1)

                    cv2.line(self.map_vis, point, a1, (0, 255, 0), 3)
                    cv2.line(self.map_vis, a1, a2, (0, 255, 0), 3)
                    cv2.line(self.map_vis, a2, point, (0, 255, 0), 3)

        # get rid of points overlapping points
        true_centers = []
        for center in centers:
            if line_map[center[1], center[0]] != 255:
                true_centers.append(center)

        centers = []
        for center in true_centers:
            n_white_pix = np.sum(line_map[center[1] - MINIMAL_WALL_DISTANCE:center[1] + MINIMAL_WALL_DISTANCE,
                                 center[0] - MINIMAL_WALL_DISTANCE:center[0] + MINIMAL_WALL_DISTANCE] == 255)
            if n_white_pix <= 0:
                centers.append(center)

        for center in centers:
            cv2.circle(self.map_vis, center, 8, (0, 0, 0), -1)

        return centers

    def get_room_openings(self, points, line_data):
        sample_size = 50
        outer_width = 50
        inner_width = 35

        room_opening_points = []

        for point in points:
            # start to iterate over pixels in circle-like structure
            line_sample = line_data[point[1] - sample_size:point[1] + sample_size,
                          point[0] - sample_size:point[0] + sample_size]
            point_sample = np.zeros(line_sample.shape, dtype=np.uint8)
            cv2.circle(point_sample, (sample_size, sample_size), outer_width, (255, 255, 255),
                       -1)  # create outer circle for masking
            cv2.circle(point_sample, (sample_size, sample_size), inner_width, (0, 0, 0),
                       -1)  # reate inner circle to later get different objects

            # mask these so you can get openings
            masked = cv2.bitwise_and(line_sample, point_sample)

            # get contours so you can get number of walls
            contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            n_walls = len(contours)
            if n_walls == 1:
                # found opening!
                room_opening_points.append(point)

        return room_opening_points

    def find_openings_using_lowest_distance(self, points):
        distances = []
        if len(points) == 0:
            print("Did not found any points, quitting because no destination was selected :(")
            exit(0)
        elif len(points) == 1:
            return points[0]
        else:
            points_copy = points.copy()
            for point in points_copy:
                points_copy.pop(0)
                for point_copy in points_copy:
                    c = round(math.sqrt(abs(point[0] - point_copy[0]) ** 2 + abs(point[1] - point_copy[1]) ** 2), 2)
                    distances.append([c, point, point_copy])

            min_num = 0
            curr_i = 0
            for i, dist in enumerate(distances):
                if i == 0:
                    min_num = dist[0]

                if dist[0] <= min_num:
                    curr_i = i

            return distances[curr_i][1:]

    def filter_points(self, points):
        # to filter out all points that are duplicit

        filtered_points = []
        for point in points:
            if point not in filtered_points:
                filtered_points.append(point)

        return filtered_points

    def calculate_path_for_drone(self, path, drone_object):
        command_data = []

        if path is None:
            print("error, path not created")
            return "not-found"

        path_list = list(path)
        for i, path_point in enumerate(path_list):
            if i == 0:
                # initial command (turning)
                init_point = path_point.coords
                upcoming_point = [350, 250]

                deg = round(math.atan((upcoming_point[0] - init_point[0]) / (upcoming_point[1] - init_point[1])) * (
                            180 / math.pi), 2)
                drone_deg = drone_object.drone_angle

                diff = deg + drone_deg
                print(diff)

                turn = "turn-left"
                if diff > 180:
                    # its better to turn right
                    turn = "turn-right"

                # else: its better to turn left

                command_data.append({"comm": turn, "val": diff})
            if i == len(path_list) - 1:
                break

            print(command_data)
        return command_data

    #
    # main code
    #
    def main(self):
        curr_drone = Drone(self.drone_angle, self.drone_pos)

        point_data, line_data, lines = self.process_map(self.map_data)
        clustered_points = self.cluster_DBSCAN(point_data)
        opening_points = self.get_room_openings(clustered_points, line_data)
        opening = self.find_openings_using_lowest_distance(opening_points)

        waypoints = self.get_waypoints_by_triangles(clustered_points, line_data)
        waypoints_filtered = self.filter_points(waypoints)

        # now to path construction
        graph_obj = Graph(waypoints_filtered, opening, drone_last_pos, lines)
        if self.debug_mode:
            graph_obj.test_plot(self.map_width, self.map_vis)
            graph_obj.matplotlib_plot(graph_obj.points_labeled)

        algorithm = CustomAStar()
        path = algorithm.astar(graph_obj.start_point, graph_obj.end_point)
        if self.debug_mode and path is not None:
            algorithm.vis_path(path, graph_obj.points_labeled)

        path_data = self.calculate_path_for_drone(path, curr_drone)
        return path_data

    def test(self):
        def on_trackbar(idx, value):
            global rho, theta_add, thresh, lines, minLineLength, maxLineGap
            self.map_vis = self.tmp_map.copy()

            if idx == 0:
                rho = value
            elif idx == 1:
                theta_add = value
            elif idx == 2:
                thresh = value
            elif idx == 3:
                lines = value
            elif idx == 4:
                minLineLength = value
            elif idx == 5:
                maxLineGap = value

            self.process_map(self.map_data, [rho, np.pi / theta_add, thresh, lines, minLineLength, maxLineGap])

        self.process_map(self.map_data)

        cv2.namedWindow('test-win')

        # Create a trackbar
        cv2.createTrackbar('rho', 'test-win', rho, 10, lambda value, idx=0: on_trackbar(idx, value))
        cv2.setTrackbarMin('rho', 'test-win', 1)
        cv2.createTrackbar('theta', 'test-win', theta_add, 360, lambda value, idx=1: on_trackbar(idx, value))
        cv2.setTrackbarMin('theta', 'test-win', 1)
        cv2.createTrackbar('thresh', 'test-win', thresh, 255, lambda value, idx=2: on_trackbar(idx, value))
        cv2.createTrackbar('minLineLength', 'test-win', minLineLength, 150,
                           lambda value, idx=3: on_trackbar(idx, value))
        cv2.createTrackbar('maxLineGap', 'test-win', maxLineGap, 150, lambda value, idx=4: on_trackbar(idx, value))

        while True:
            cv2.imshow('test-win', self.map_vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # Escape key to exit
                break


class Point:
    def __init__(self, coords, id):
        self.coords = coords
        self.id = id

        # going to put some of the leading points here
        self.leading_to = []


class Graph:
    def __init__(self, points, dest_points, drone_pos, lines):

        self.VERTICAL_PASS_COEF = 2
        self.MIN_POINT_DIST = 30

        self.path_array = []
        self.points_labeled = []

        start_point_coord = drone_pos
        end_point_coord = dest_points

        # self.start_point = Point(start_point_coord, 0)
        # self.end_point = Point(end_point_coord, 1)

        self.all_points = []

        self.all_points.append(start_point_coord)
        self.all_points.extend(points)
        self.all_points.append(end_point_coord)

        # assign id to every point
        id = 0
        for point in self.all_points:
            self.points_labeled.append(Point(point, id))
            id += 1

        points_copy = self.all_points.copy()
        while True:
            if len(points_copy) == 0:
                break

            point = points_copy[0]
            points_copy.pop(0)
            for point2 in points_copy:
                collision = False
                point_path = [point, point2]

                # iterate on every line to check colision
                for line in lines:
                    line_path = [line[0][:2].tolist(), line[0][2:].tolist()]

                    collision = self.__check_for_collision__(point_path, line_path)
                    if collision:
                        # did collide together
                        break

                if not collision:
                    # did not collide!

                    # search all the points and assign them leading to points

                    #
                    # This is the ugliest piece of code I have ever written in my programming history, I am deeply sorry for all my colleagues that actually have to see this big pile of non-optimised code
                    # TODO get rid of this 6-for-loop situation

                    for dest_point in self.points_labeled:
                        if dest_point.coords[0] == point[0] and dest_point.coords[1] == point[1]:
                            for dest_point2 in self.points_labeled:
                                if dest_point2.coords[0] == point2[0] and dest_point2.coords[1] == point2[1]:
                                    dest_point.leading_to.append(dest_point2)

        self.start_point = self.points_labeled[0]
        self.end_point = self.points_labeled[len(self.points_labeled) - 1]

    def __check_segment__(self, p, q, r):
        if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
                (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    def __check_orientation__(self, p, q, r):
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            # Clockwise orientation 
            return 1
        elif (val < 0):
            # Counterclockwise orientation 
            return 2
        else:
            # Collinear orientation 
            return 0

    def __check_for_collision__(self, line_1, line_2):
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

    def test_plot(self, map_width, map_vis):
        plot_img = np.full((map_width, map_width, 3), 255, dtype='uint8')

        for point in self.points_labeled:
            cv2.circle(plot_img, point.coords, 8, (125, 125, 0), -1)
            if point.id != self.start_point.id and point.id != self.end_point.id:
                cv2.putText(plot_img, str(point.id), [point.coords[0] + 5, point.coords[1] - 5], font, fontScale,
                            (255, 0, 0), 1, cv2.LINE_AA)

                # start point
        cv2.circle(plot_img, self.start_point.coords, 10, (0, 255, 0), -1)  # start for green
        cv2.putText(plot_img, 'Start point', (self.start_point.coords[0] + 10, self.start_point.coords[1] - 10), font,
                    fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

        # end point
        cv2.circle(plot_img, self.end_point.coords, 10, (0, 0, 255), -1)  # end for red
        cv2.putText(plot_img, 'End point', (self.end_point.coords[0] + 10, self.end_point.coords[1] - 10), font,
                    fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imshow("plot_img", plot_img)
        cv2.imshow("map_vis", map_vis)
        cv2.waitKey(0)

        cv2.destroyWindow("plot_img")
        cv2.destroyWindow("map_vis")

    def matplotlib_plot(self, waypoints):
        G = nx.Graph()

        point_ids = []
        for point in waypoints:
            # add edge to this waypoint
            if point.id == 78:
                pass

            if len(point.leading_to) > 0:
                point_ids.append(str(point.id))

            for leading_point in point.leading_to:
                G.add_edge(str(point.id), str(leading_point.id))

        pos = nx.spring_layout(G)  # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, nodelist=point_ids)  # waypoints

        if len(self.end_point.leading_to) > 0:
            nx.draw_networkx_nodes(G, pos, node_size=1400, nodelist=[str(self.start_point.id), str(self.end_point.id)],
                                   node_color='blue')  # start and stop points
        else:
            print("no connection found for destination")
            nx.draw_networkx_nodes(G, pos, node_size=1400, nodelist=[str(self.start_point.id), str(point_ids[-1])],
                                   node_color='blue')

            # edges
        nx.draw_networkx_edges(G, pos, width=6)

        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.show()


def read_data(filename):
    file = open(os.path.join(os.path.dirname(os.path.abspath(__file__))[:-3], "src/Flight_logs/scan_data/" + filename),
                'r')
    data = file.read()
    data = eval(data)

    return data


if __name__ == "__main__":
    # variables
    map_width = 700

    data = read_data("perfecto_room.txt")

    drone_last_pos = (round(map_width / 2), round(map_width / 2))  # TODO pak změň - teď je to default
    drone_angle = 0
    debug = True

    proc_instance = MapProcessing(inp_data=data, map_shape=(map_width, map_width), drone_pos=drone_last_pos,
                                  drone_angle=drone_angle, debug=debug)
    # data = proc_instance.main()
    proc_instance.test()
