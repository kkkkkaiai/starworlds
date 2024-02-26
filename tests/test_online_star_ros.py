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
from dynamic_system.modulation import modulation_velocity

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Polygon, PolygonStamped
from copy import copy

import numpy as np
import matplotlib.pyplot as plt


robot_c = None
robot_yaw = None
robot_v = None
robot_w = None
is_robot_init = False

laser = None
laser_points = []
laser_points_filter_max_range = []
is_laser_init = False
resolusion = 0.05

def odom_cb(msg):
    global robot_c, robot_yaw, is_robot_init, robot_v, robot_w
    robot_c = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    robot_yaw = np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)*2
    robot_v = msg.twist.twist.linear.x
    robot_w = msg.twist.twist.angular.z
    if not is_robot_init:
        is_robot_init = True

def laser_callback(msg):
    global laser_points, laser_points_filter_max_range
    laser_points = []
    laser_points_filter_max_range = []
    inf = float('inf')
    is_max_data = False
    for i in range(len(msg.ranges)):
        # check if the point is inf
        point_data = msg.ranges[i]
        if point_data == inf:
            is_max_data = True
            point_data = msg.range_max - 0.0001

        if point_data < msg.range_max and point_data > msg.range_min:
            angle = msg.angle_min + i * msg.angle_increment
            x = point_data * np.cos(angle)
            y = point_data * np.sin(angle)
            laser_points.append([x, y])
            if not is_max_data:
                laser_points_filter_max_range.append([x, y])

        is_max_data = False
        is_laser_init = True

def transform_to_global_frame(points, robot_c, robot_yaw):
    points = np.array(points)
    points = points.T
    points = np.vstack((points, np.ones(points.shape[1])))
    transform_matrix = np.array([[np.cos(robot_yaw), -np.sin(robot_yaw), robot_c[0]], 
                  [np.sin(robot_yaw), np.cos(robot_yaw), robot_c[1]], 
                  [0, 0, 1]])
    points = np.dot(transform_matrix, points)
    return points[:2].T


class PID:
    def __init__(self, kp_v=0.5, kd_v=0.1, kp_w=0.6, kd_w=0.2) -> None:
        self.kp_v = kp_v
        self.kd_v = kd_v

        self.kp_w = kp_w
        self.kd_w = kd_w

        self.v_ref = 0.34
        self.w_ref = 0

        self.prev_v_error = 0
        self.prev_w_error = 0

    def set_v_ref(self, v_ref):
        self.v_ref = v_ref

    def cal_ref(self, current_state, target_state):
        # current_state: x, y, theta, v, w
        # target_state: x, y, theta
        theta_error = target_state[2] - current_state[2]
        if theta_error > np.pi:
            theta_error -= 2 * np.pi
        if theta_error < -np.pi:
            theta_error += 2 * np.pi
        
        # if theta_error is large, the ref_v should decrease
        ref_v = self.v_ref * (0.5/np.abs(theta_error))
        ref_v = np.min([ref_v, self.v_ref])
        ref_w = theta_error

        return ref_v, ref_w

    def update(self, current_state, target_state):
        # current_state: x, y, theta, v, w
        # target_state: x, y, theta
        ref_v, ref_w = self.cal_ref(current_state, target_state)
        v_error = ref_v - current_state[3]
        w_error = ref_w

        p_term_v = self.kp_v * v_error
        d_term_v = self.kd_v * (v_error - self.prev_v_error)

        p_term_w = self.kp_w * w_error
        d_term_w = self.kd_w * (w_error - self.prev_w_error)

        self.prev_v_error = v_error
        self.prev_w_error = w_error

        return p_term_v + d_term_v, p_term_w + d_term_w


def publish_cmd_vel(pub, control_signal):
    cmd = Twist()
    cmd.linear.x = control_signal[0]
    cmd.angular.z = control_signal[1]
    pub.publish(cmd)

def publish_arrow_marker(pub, position, yaw, id):
    marker = Marker()
    marker.header.frame_id = "odom"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "basic_shapes"
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = np.sin(yaw/2)
    marker.pose.orientation.w = np.cos(yaw/2)
    marker.scale.x = 0.5
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    pub.publish(marker)


if __name__ == '__main__':

    rospy.init_node('test_star_ros', anonymous=True)
    print('test_online_star_ros')

    odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb)
    laser_sub = rospy.Subscriber('/scan', LaserScan, laser_callback)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    arrow_pub = rospy.Publisher('/arrow_marker', Marker, queue_size=10)

    xlim = [-5, 20]
    ylim = [-5, 20]

    # start_position = np.array([8.0, 5.5])
    rate = rospy.Rate(10)
    while not is_robot_init and not is_laser_init:
        print('temp')
        rate.sleep()

    pid_controller = PID()

    # to be assigned
    # case 1
    # start_position = np.array([0.0, 0.0])
    # goal_position = np.array([1.1, 3.7])
    # case 2
    start_position = robot_c
    # goal_position = np.array([4.7, 1.23])
    goal_position = np.array([-2, 3])
    # goal_position = np.array([-1.5, 0.5])


    laser_points = transform_to_global_frame(laser_points, robot_c, robot_yaw)
    laser_points_filter_max_range = transform_to_global_frame(laser_points_filter_max_range, robot_c, robot_yaw)
    # no_yaw_laser_points = transform_to_global_frame(laser_points, robot_c, 0)
    # print(laser_points.T)
    # print(len(laser_points))

    star_node = StarshapedRep(laser_points, start_position, 
                              robot_theta=robot_yaw, robot_radius=0.5, ros_simu=True, 
                              points_for_frontier=laser_points_filter_max_range)
    star_node.draw('test1')
    graph_manager = GraphManager(star_node)
    start_star_id = copy(graph_manager.start_id)
    current_star_id = copy(graph_manager.start_id)

    path, reach = graph_manager.find_path(current_star_id, goal_position)
    print(path)
    local_id = path['path_id'][-1]
    local_position = path['path'][-1]

    reach_local = False
    reach_global = False
    update_hz = 10
    iter = 0
    interval = 2

    local_position_limit = 0.2

    rate = rospy.Rate(update_hz)

    current_position = copy(start_position)
    while not reach_global and iter < 10000:
        iter += 1
        new_velocity = modulation_velocity(current_position, local_position, graph_manager)
        new_velocity *= 0.5
        new_position = current_position + new_velocity
        target_yaw = np.arctan2(new_velocity[1], new_velocity[0])
        # print('position', current_position, new_position, local_position)

        current_state = np.array([robot_c[0], robot_c[1], robot_yaw, robot_v, robot_w])
        target_state = np.array([new_position[0], new_position[1], target_yaw])
        control_signal = pid_controller.update(current_state, target_state)
        # print('control_signal', control_signal)
        current_position = robot_c

        # judge if reach the local position
        if np.linalg.norm(local_position - current_position) < local_position_limit and \
            np.linalg.norm(local_position - goal_position) > 0.001:
            print('reach the local position')
            reach_local = True

        # judge if reach the global position
        if np.linalg.norm(goal_position - current_position) < 0.1:
            print('reach the global position')
            reach_global = True
            reach_local = False

        # if reach local, generate a new star node and find a new path
        if reach_local:
            print('test', local_position, new_position)
            # generate a new star node
            laser_points = transform_to_global_frame(laser_points, robot_c, robot_yaw)
            laser_points_filter_max_range = transform_to_global_frame(laser_points_filter_max_range, robot_c, robot_yaw)

            find_obstacle = False
            in_other_starshape = False
            # judge if there are points around the local goal
            for i in range(len(laser_points)):
                if np.linalg.norm(laser_points[i] - local_position) < 0.2:
                    print('remove the obstacle', np.linalg.norm(laser_points[i] - local_position))
                    find_obstacle = True
                    # remove the node in the graph
                    graph_manager.remove_node(id)
                    # generate a new path
                    path, reach = graph_manager.find_path(current_star_id, goal_position)
                    id = path['path_id'][-1]
                    local_position = path['path'][-1]
                   
                    break

            if not find_obstacle:
                print('extend obstacle')
                # laser_points = laser_points.tolist()
                star_rep = StarshapedRep(laser_points, current_position, robot_theta=robot_yaw, robot_radius=0.5,
                                         ros_simu=True, points_for_frontier=laser_points_filter_max_range)
                graph_manager.extend_star_node(local_id, star_rep)
                current_star_id = copy(local_id)
                star_rep.draw('test1')
                path, reach = graph_manager.find_path(current_star_id, goal_position)
                local_id = path['path_id'][-1]
                local_position = path['path'][-1]
            
            reach_local = False

        # publish current position to new position
        publish_arrow_marker(arrow_pub, current_position, target_yaw, 0)
        publish_cmd_vel(cmd_pub, control_signal)
        plt.axis('equal')
        plt.savefig('finding_path_ros.png', dpi=300)
        rate.sleep()

    publish_cmd_vel(cmd_pub, [0, 0])

   
