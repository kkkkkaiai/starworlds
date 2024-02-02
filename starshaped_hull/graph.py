import numpy as np

from collections import defaultdict, deque
from starshaped_hull.starshaped_fit import StarshapedRep

class Node:
    def __init__(self, point, type=1) -> None:
        self._point = point
        self._type = type # 0 is starshaped node, 1 is frontier node
        self._valid = True

    def __hash__(self) -> int:
        return hash(tuple(self._point))

    def stuck(self):
        self._valid = False

    def is_stuck(self):
        return self._valid
    
    def star_rep(self, star_rep):
        self._star_rep = star_rep
        self._type = 0

# construct undirect graph
class GraphManager:
    def __init__(self, root: StarshapedRep) -> None:
        self._edges = defaultdict(list)
        self._nodes = defaultdict(Node)
        
        self._id_list = []
        self.start_id = None
        self.id = 0
        self.is_init = False
        self.initial(root)

    def gen_id(self):
        self.id += 1
        return self.id

    def initial(self, root):
        self.start_id = gen_id()
        self._nodes[self.start_id] = Node(root.center)
        frontier_points = root.frontier_points
        for i in range(len(frontier_points)):
            new_id = gen_id()
            self.add_node(self.start_id, new_id)
            self._nodes[new_id] = Node(frontier_points[i])

        self.initial = True

    def find_path(self, goal_point):
        # return subgoal's id
        if not self.initial:
            return None

    def extend_node(self, node_id, star_rep):
        self._nodes[node_id].star_rep(star_rep)
        frontier_points = star_rep.frontier_points
        for i in range(len(frontier_points)):
            new_id = gen_id()
            self.add_node(node_id, new_id)
            self._nodes[new_id] = Node(frontier_points[i])

    def invalid_node(self, node_id):
        self._nodes[node_id].stuck()

    def get_neighbor(self, node_id):
        return self._edges[node_id]

    def add_node(self, node_id, neighbor_id):
        self._edges[node_id].append(neighbor_id)
        self._edges[neighbor_id].append(node_id)



    
