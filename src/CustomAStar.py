from astar import AStar
import math
import cv2
import numpy as np

MAP_WIDTH = 1000

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1

class CustomAStar(AStar):
    def __init__(self):
        pass
        # self.nodes = nodes

        # self.nodes = {}
        #
        # #process all points to supported format
        # for node in nodes:
        #     leading_to = []
        #     for lead_point in node.leading_to:
        #         lead_id = lead_point.id
        #         lead_dist = round(self.distance_between(node, lead_point), 2)
        #         leading_to.append((lead_id, lead_dist))
        #
        #     self.nodes[node.id] = leading_to
        #
        # print(self.nodes)

    def neighbors(self, n):
        for n1 in n.leading_to:
            yield n1

    def distance_between(self, n1, n2):
        dx = n1.coords[0] - n2.coords[0]
        dy = n1.coords[1] - n2.coords[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def heuristic_cost_estimate(self, current, goal):
        dx = current.coords[0] - goal.coords[0]
        dy = current.coords[1] - goal.coords[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def is_goal_reached(self, current, goal):
        return current.id == goal.id
    
    #
    # Other funcs
    #

    def vis_path(self, path, all_points):
        plot_img = np.full((MAP_WIDTH, MAP_WIDTH, 3), 255, dtype='uint8')

        for point in all_points:
            cv2.circle(plot_img, point.coords, 8, (0, 0, 0), 3)
            cv2.circle(plot_img, point.coords, 8, (125, 125, 0), -1)

        path_list = list(path)
        for i, path_point in enumerate(path_list):
            if i != len(path_list) - 1:
                cv2.line(plot_img, path_point.coords, path_list[i + 1].coords, (0, 255, 0), 2)
            
            cv2.circle(plot_img, path_point.coords, 10, (128, 0, 128), -1)
            cv2.putText(plot_img, str(path_point.id), [point.coords[0] + 5, path_point.coords[1] - 5], font, fontScale, (255, 0, 0), 1, cv2.LINE_AA) 
            print(path_point.id)

        cv2.imshow("result", plot_img)
        cv2.waitKey(0)