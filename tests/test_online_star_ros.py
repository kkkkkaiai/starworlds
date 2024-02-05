#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from planner.trajectory_generator.spline_interpolate import *
from starshaped_hull.graph import GraphManager
from obstacles import Frame,  StarshapedPolygon
from sensors.laser_anyshape import Laser
from sklearn.cluster import DBSCAN
from starshaped_hull.starshaped_fit import StarshapedRep

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np
import matplotlib.pyplot as plt


robot_c = None
robot_yaw = None
is_robot_init = False

laser = None
is_laser_init = False
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

def odom_cb(msg):
    global robot_c, robot_yaw, is_robot_init
    robot_c = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    robot_yaw = np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)*2
    if not is_robot_init:
        is_robot_init = True

def laser_callback(msg):
    global laser_points
    laser_points = []
    inf = float('inf')
    for i in range(len(msg.ranges)):
        # check if the point is inf
        if msg.ranges[i] == inf:
            continue

        if msg.ranges[i] < msg.range_max and msg.ranges[i] > msg.range_min:
            angle = msg.angle_min + i * msg.angle_increment
            x = msg.ranges[i] * np.cos(angle)
            y = msg.ranges[i] * np.sin(angle)
            laser_points.append([x, y])

def transform_to_global_frame(points, robot_c, robot_yaw):
    points = np.array(points)
    points = points.T
    points = np.vstack((points, np.ones(points.shape[1])))
    T = np.array([[np.cos(robot_yaw), -np.sin(robot_yaw), robot_c[0]], 
                  [np.sin(robot_yaw), np.cos(robot_yaw), robot_c[1]], 
                  [0, 0, 1]])
    points = np.dot(T, points)
    return points[:2].T
    

def starshaped_polygon(points, point_num=200, plot=False):
    xlim = [-5, 100]
    ylim = [-5, 150]
    pol = StarshapedPolygon(points);

    while True:
        x = np.array([np.random.uniform(*xlim), np.random.uniform(*ylim)])
        if pol.exterior_point(x):
            break
    b = pol.boundary_mapping(x)
    n = pol.normal(x)
    tp = pol.tangent_points(x)
    dir = pol.reference_direction(x)

    if plot:
        _, ax = pol.draw()
        ax.plot(*zip(pol.xr(Frame.GLOBAL), x), 'k--o')
        if b is not None:
            ax.plot(*b, 'y+')
            ax.quiver(*b, *n)
        if tp:
            ax.plot(*zip(x, tp[0]), 'g:')
            ax.plot(*zip(x, tp[1]), 'g:')
        ax.quiver(*pol.xr(Frame.GLOBAL), *dir, color='c', zorder=3)

    b_list = []

    for i in np.linspace(0, 2 * np.pi, point_num):
        x = pol.xr() + 100*np.array([np.cos(i), np.sin(i)])
        b = pol.boundary_mapping(x)
        b_list.append(b)
        n = pol.normal(b)
        if plot:
            ax.quiver(*b, *n, color='g')
    if plot:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.savefig("test_starshaped_polygon.png", dpi=300)

    b_list = np.array(b_list).T

    return b_list



if __name__ == '__main__':
    print('before')
    rospy.init_node('test_star_ros', anonymous=True)
    print('test_online_star_ros')
    # get odom 
    odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb)
    # get laser scan
    laser_sub = rospy.Subscriber('/scan', LaserScan, laser_callback)

    xlim = [-5, 20]
    ylim = [-5, 20]

    # start_position = np.array([8.0, 5.5])
    rate = rospy.Rate(10)
    while not is_robot_init:
        print('temp')
        rate.sleep()

    # to be assigned
    goal_position = np.array([8, 5.0])
    
    start_position = robot_c

    laser_points = transform_to_global_frame(laser_points, robot_c, robot_yaw)

    print(laser_points.T)
    print(len(laser_points))

    # b_list = starshaped_polygon(laser_points, plot=False)
    # print(b_list)
    # star_ds(b_list*resolusion, start_position, start_position, x_lim=xlim, y_lim=ylim, plot=False)
    # star_ds(laser_points, start_position)
    # plt.cla()
    star_node = StarshapedRep(laser_points.T, start_position)
    star_node.draw('test1')
    graph_manager = GraphManager(star_node)
    path, reach = graph_manager.find_path(1, goal_position)
    id = path['path_id'][-1]

    plot_number = 0
    plt.savefig('finding_path1.png', dpi=300)
