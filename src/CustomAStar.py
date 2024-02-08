from astar import AStar
import math


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