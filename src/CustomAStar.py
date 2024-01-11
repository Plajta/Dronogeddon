from astar import AStar


class CustomAStar(AStar):
    def __init__(self, nodes):
        self.nodes = nodes

    def neighbors(self, n):
        for n1, d in self.nodes[n]:
            yield n1

    def distance_between(self, n1, n2):
        for n, d in self.nodes[n1]:
            if n == n2:
                return d

    def heuristic_cost_estimate(self, current, goal):
        return 1

    def is_goal_reached(self, current, goal):
        return current == goal