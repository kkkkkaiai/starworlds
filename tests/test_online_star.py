
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from envs.gridmap import OccupancyGridMap
from planner.path_generator.astar import a_star
from planner.trajectory_generator.spline_interpolate import *

from sensors.laser_anyshape import Laser
from sklearn.cluster import DBSCAN
from test_starshaped_polygon import star_ds, starshaped_polygon

import numpy as np
import matplotlib.pyplot as plt

resolusion = 0.05

class PathManager:
    def __init__(self, gridmap, start_node=(6.0, 3.0), end_node=(16.0, 16.0)) -> None:
        self._map = gridmap
        self._resolution = self._map.cell_size
        self._start_node = start_node
        self._end_node = end_node
        self._path = None
        self._path_px = None

    def find_path(self, start_node, end_node):
        # run A*
        self._start_node = start_node
        self._end_node = end_node
        self._path, self._path_px = a_star(self._start_node, self._end_node, self._map, movement='8N')
        self._path = np.asarray(self._path) / self._resolution
        return self._path
    
    def find_gradient(self):
        '''
        find the index of which the gradient is larger than a threshold
        '''
        path_index = []
        for i in range(1, len(self._path)-1):
            if abs(np.linalg.norm(self._path[i]-self._path[i-1]) - \
                   np.linalg.norm(self._path[i+1]-self._path[i])) > 0.2:
                path_index.append(i-1)
        path_index.append(len(self._path)-1)
        return self._path[path_index]
    
    def is_path_found(self):
        return self._path is not None
    
    def spline_interpolate(self, path=None, ds=0.1):
        if path is None:
            path = self._path
        cx, cy, cyaw, ck, s = calc_spline_course(path[:, 0]*self._resolution, path[:, 1]*self._resolution, ds=ds)
        self._sp = calc_speed_profile(cx, cy, cyaw, 10)
        self._ref_path = PATH(cx, cy, cyaw, ck)
        print('Path length: ', len(path), 'Interpolated path length: ', self._ref_path.length)
        return self._ref_path, self._sp

    def plot_map(self):
        self._map.plot()

    def plot_path(self):
        if self.is_path_found():
            plot_path(self._path, linestyle='--', label='origin path')
        else:
            # plot start and goal points over the map (in pixels)
            start_node_px = self._map.get_index_from_coordinates(self._start_node[0], self._start_node[1])
            goal_node_px = self._map.get_index_from_coordinates(self._end_node[0], self._end_node[1])

            plt.plot(start_node_px[0], start_node_px[1], 'ro')
            plt.plot(goal_node_px[0], goal_node_px[1], 'go')
            raise ValueError('Path is not found')
        
    def plot_interpolated_path(self):
        try:
            if self._ref_path is not None:

                plot_path(np.array([np.array(self._ref_path.cx)/self._resolution, np.array(self._ref_path.cy)/self._resolution]).T, \
                          color='cyan', label='interpolated path')
        except:
            print('Not do interpolation yet, please call spline_interpolate() first')


if __name__ == '__main__':
    map_file = 'maps/obstacle_map_occupancy.png'
    gridmap = OccupancyGridMap.from_png(map_file, 0.1)
    pM = PathManager(gridmap)
    start_position = np.array([6.0, 10])

    robot_c = start_position
    robot_yaw = 0.0
    
    laser = Laser(beams=128)
    laser.set_map(gridmap)
    de_obs = laser.state2obs(robot_c, robot_yaw, False)

    laser_points = de_obs['point']

    pM.plot_map()
    # plot laser points
    plt.scatter(laser_points[:, 0], laser_points[:, 1], s=1, c='red', label='laser points')
    plt.savefig('test_laser_points.png', dpi=300)

    # convert laser points' type to list
    laser_points = laser_points.tolist()

    points = []
    for i in range(len(laser_points)):
        points.append(tuple(laser_points[i]))

    plt.cla()
    starshaped_polygon(laser_points)
    # star_ds(laser_points, start_position)