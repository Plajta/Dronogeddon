from astar import AStar
import math


class CustomAStar(AStar):
    def __init__(self, nodes):
        self.nodes = nodes

    def neighbors(self, n):
        for n1 in n.leading_to:
            yield n1

    def distance_between(self, n1, n2):
        dx = n1.coord[0] - n2.coord[0]
        dy = n1.coord[1] - n2.coord[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def heuristic_cost_estimate(self, current, goal):
        dx = current.coord[0] - goal.coord[0]
        dy = current.coord[1] - goal.coord[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def is_goal_reached(self, current, goal):
        return current.id == goal.id