import numpy as np
import heapq
import sys

from collections import defaultdict, deque
from starshaped_hull.starshaped_fit import StarshapedRep

class Node:
    def __init__(self, point, type=1) -> None:
        self._point = point
        self._type = type # 0 is starshaped node, 1 is frontier node
        self._valid = True
        self._star_rep_id = None
        self._star_rep = None

    def __hash__(self) -> int:
        return hash(tuple(self._point))

    def stuck(self):
        self._valid = False

    def is_stuck(self):
        return not self._valid

    def is_starshape(self):
        return self._type == 0

    def star_id(self):
        if self.is_starshape() and self._star_rep_id is not None:
            return self._star_rep_id
        return None
    
    def get_gamma(self, point):
        return self._star_rep.get_gamma(point)

    def star_rep(self, star_rep, id):
        self._star_rep = star_rep
        self._type = 0
        self._star_rep_id = id
        
    def is_in_star(self, point):
        if self._type == 0:
            return self._star_rep.is_in_star(point)
        return False

# construct undirect graph
class GraphManager:
    def __init__(self, root: StarshapedRep) -> None:
        self._edges = defaultdict(list)
        self._nodes = defaultdict(Node)
        
        self._id_list = []
        self._star_id_list = []
        self.start_id = None
        self.id = 0
        self._star_rep_id = 0
        self.is_init = False
        self.initial(root)

    def gen_id(self):
        self.id += 1
        return self.id

    def initial(self, root):
        self.start_id = self.gen_id()
        self._star_id_list.append(self.start_id)
        self._nodes[self.start_id] = Node(root.center)
        self._nodes[self.start_id].star_rep(root, self.start_id)
        frontier_points = root.frontier_points
        for i in range(len(frontier_points)):
            new_id = self.gen_id()
            self.add_node(self.start_id, new_id)
            self._nodes[new_id] = Node(frontier_points[i])

        self.initial = True

    def find_path(self, in_star_id, goal_point):
        # in_star_id is the id in current starshaped polygon
        # return subgoal's point as attractive point
        if not self.initial or in_star_id not in self._nodes:
            return None
        
        class cell:
            def __init__(self):
                self.parent = -1
                self.g = float('inf')

        road_map = defaultdict(cell)
        road_map[in_star_id].g = 0

        # traverse all the node in the graph and find the nearest node to the goal point without stuck
        nearest_node_id = None
        for node_id in self._nodes:
            if self._nodes[node_id].is_stuck():
                continue
            if nearest_node_id is None:
                nearest_node_id = node_id
            elif np.linalg.norm(goal_point - self._nodes[node_id]._point) < np.linalg.norm(goal_point - self._nodes[nearest_node_id]._point):
                nearest_node_id = node_id

        if nearest_node_id is None:
            # exit with a message
            print('No valid nearest node to the goal point in the graph')
            exit()

        open_list = []
        heapq.heappush(open_list, (0, in_star_id))
        close_list = set()
        
        found_path = False
        reach_goal = False

        if self._nodes[in_star_id].is_in_star(goal_point):
            road_map[in_star_id].parent = -1
            close_list.add(in_star_id)
            found_path = True
            reach_goal = True
        else:   
            while open_list and not found_path:
                _, cur_id = heapq.heappop(open_list)
                if cur_id in close_list:
                    continue

                close_list.add(cur_id)

                neighbor_ids = self.get_neighbor(cur_id)   
                for next_id in neighbor_ids:
                    if self._nodes[next_id].is_in_star(goal_point):
                        road_map[next_id].parent = cur_id
                        found_path = True
                        reach_goal = True
                    
                    if self._nodes[next_id].is_stuck():
                        continue

                    if next_id == nearest_node_id:
                        found_path = True
                        close_list.add(next_id)

                    if road_map[next_id].g > road_map[cur_id].g + \
                            np.linalg.norm(np.array(self._nodes[next_id]._point) - np.array(self._nodes[cur_id]._point)):
                        road_map[next_id].g = road_map[cur_id].g + \
                                                np.linalg.norm(np.array(self._nodes[next_id]._point) - np.array(self._nodes[cur_id]._point))

                        f_value = road_map[next_id].g + np.linalg.norm(goal_point - self._nodes[next_id]._point)
                        heapq.heappush(open_list, (f_value, next_id))
                        road_map[next_id].parent = cur_id

        path = {'path':[], 'path_id':[]}
        if found_path:
            # if goal point in the star, return the goal point
            if len(close_list) == 1:
                # log path and path id
                print('in')
                path['path'] = [self._nodes[in_star_id]._point, goal_point]
                path['path_id'] = [in_star_id, 0] # 0 is the id of the goal point
            else:
                backward_path = {'path':[self._nodes[nearest_node_id]._point], 'path_id':[nearest_node_id]}
                node_id = road_map[nearest_node_id].parent
                while node_id != -1:
                    backward_path['path'].append(self._nodes[node_id]._point)
                    backward_path['path_id'].append(node_id)
                    node_id = (road_map[node_id]).parent

                path['path'] = backward_path['path'][::-1]
                path['path_id'] = backward_path['path_id'][::-1]

            print("[INFO] PRM: Found a path! Path length is {}. ".format(len(path)))
        else:
            print('[WARNNING] PRM: Was not able to find a path!')

        return path, reach_goal

    def extend_node(self, node_id, star_rep):
        # node_id is the id of current node
        self._nodes[node_id].star_rep(star_rep, node_id)
        frontier_points = star_rep.frontier_points
        for i in range(len(frontier_points)):
            # check the frontier points is in the other starshaped polygon
            for j in range(len(self._star_id_list)):
                if self._nodes[self._star_id_list[j]].is_in_star(frontier_points[i]):
                    continue
            new_id = self.gen_id()
            self.add_node(node_id, new_id)
            self._nodes[new_id] = Node(frontier_points[i])
        
        self._star_id_list.append(node_id)
            
    def get_star_id(self, node_id):
        return self._nodes[node_id].star_id()

    def invalid_node(self, node_id):
        self._nodes[node_id].stuck()

    def get_neighbor(self, node_id):
        return self._edges[node_id]

    def add_node(self, node_id, neighbor_id):
        self._edges[node_id].append(neighbor_id)
        self._edges[neighbor_id].append(node_id)



    
