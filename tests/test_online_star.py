
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.animation import FuncAnimation
from envs.gridmap import OccupancyGridMap
from planner.path_generator.astar import a_star
from planner.trajectory_generator.spline_interpolate import *
from starshaped_hull.graph import GraphManager
from dynamic_system.modulation import modulation_velocity

from sensors.laser_anyshape import Laser
from sklearn.cluster import DBSCAN
from test_starshaped_polygon import star_ds, starshaped_polygon
from starshaped_hull.starshaped_fit import StarshapedRep
from typing import Optional
from copy import copy

import warnings
import time
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import numpy.typing as npt


def evaluation_modulation_result(x_lim, y_lim, graph_manager, local_position):
    # draw the starshaped polygon
    plt.cla()
    starshaped_ids = graph_manager._star_id_list
    for i in range(len(starshaped_ids)):
        star_id = starshaped_ids[i]
        star = graph_manager._nodes[star_id]._star_rep
        star.draw('test'+str(i))

    x = np.arange(x_lim[0], x_lim[1]+5, 0.15)
    y = np.arange(y_lim[0], y_lim[1]+5, 0.15)
    print(len(x), len(y))
    for i in range(len(x)):
        for j in range(len(y)):
            cur_position = np.array([x[i], y[j]])
            # if the gamma is less than 1, continue
            # if graph_manager._nodes[starshaped_ids[0]]._star_rep.get_gamma(cur_position) < 1:
            #     continue

            new_velocity, _ = modulation_velocity(cur_position, local_position, graph_manager)
            # normal
            new_velocity = new_velocity / np.linalg.norm(new_velocity) *0.1

            plt.arrow(cur_position[0], cur_position[1], new_velocity[0], new_velocity[1], head_width=0.01, head_length=0.01, fc='black', ec='black')
    plt.axis('equal')
    plt.savefig('evaluation_result.png', dpi=300)


def modulation_round_robot(position, local_position, graph_manager, safety_margin=0.5):
    # calculate the gamma of current position
    current_velocity, current_min_Gamma = modulation_velocity(position, local_position, graph_manager)
    current_velocity = current_velocity / np.linalg.norm(current_velocity)

    # front, left, right point
    front_point = position + current_velocity * safety_margin
    left_point = position + np.array([-current_velocity[1], current_velocity[0]]) * safety_margin
    right_point = position + np.array([current_velocity[1], -current_velocity[0]]) * safety_margin
    print('point', front_point, left_point, right_point)

    # the gamma of the front, left and right point
    front_velocity, front_min_Gamma = modulation_velocity(front_point, local_position, graph_manager)
    left_velocity, left_min_Gamma = modulation_velocity(left_point, local_position, graph_manager)
    right_velocity, right_min_Gamma = modulation_velocity(right_point, local_position, graph_manager)
    
    if current_min_Gamma == np.inf:
        current_min_Gamma = 1e10
    if front_min_Gamma == np.inf:
        front_min_Gamma = 1e10
    if left_min_Gamma == np.inf:
        left_min_Gamma = 1e10
    if right_min_Gamma == np.inf:
        right_min_Gamma = 1e10

    # computing weight based on the Gamma value
    weight_temp = np.array([front_min_Gamma, left_min_Gamma, right_min_Gamma, current_min_Gamma])
    # filter the weight with the lower of 1.05
    weight = copy(weight_temp)
    weight_import = np.where(weight_temp <= 1.05)
    print(weight_import, weight)
    weight_temp = np.zeros(4)
    if len(weight_import[0]) != 0:
        for i in range(len(weight_import[0])):
            weight_temp[weight_import[0][i]] = weight[weight_import[0][i]]
    else:
        weight_temp = copy(weight)
    print(weight_temp)
    weight = weight_temp / np.sum(weight_temp)
    
    # calculate the new velocity
    new_velocity = front_velocity * weight[0] + left_velocity * weight[1] + right_velocity * weight[2] + current_velocity * weight[3]
     
    return new_velocity


if __name__ == '__main__':
    # map params
    map_file = 'maps/obstacle_map_occupancy_2.png'
    resolusion = 0.1
    xlim = [-1, 11]
    ylim = [-1, 17]
    
    # load map
    gridmap = OccupancyGridMap.from_png(map_file, resolusion)

    # init robot
    start_position = np.array([8, 3])
    goal_position = np.array([9, 13])
    robot_c = start_position
    robot_yaw = 0.0
    
    # init laser
    laser = Laser(beams=512, laser_length=10)
    laser.set_map(gridmap)
    laser_detect_range = laser.max_detect_distance()

    # plot params
    plt.cla()
    fig = plt.figure(figsize=(10, 10))
    xlim = [-1, 7]
    ylim = [-1, 10]
    plt.xlim(xlim)
    plt.ylim(ylim)

    # plot start position and goal position with five-pointed star
    plt.scatter(start_position[0], start_position[1], c='r', s=200, marker='*', zorder=5)
    plt.scatter(goal_position[0], goal_position[1], c='g', s=200, marker='*', zorder=5)

    # plot a sparse map
    # for i in range(xlim[0], xlim[1]):
    #     for j in range(ylim[0], ylim[1]):
    #         if gridmap.is_occupied((i, j)):
    #             plt.plot(i, j, 'k.')

    # generate laser points and starshaped representation
    de_obs = laser.state2obs(robot_c, robot_yaw, False)
    laser_points = de_obs['point']
    # filter max detect range
    # print(np.linalg.norm(laser_points-robot_c/resolusion, axis=1) < laser_detect_range / resolusion)
    # laser_points = laser_points[np.linalg.norm(laser_points-robot_c/resolusion, axis=1) < laser_detect_range / resolusion, :]
    # print(laser_points)

    star_rep = StarshapedRep((laser_points*resolusion), start_position, robot_radius=0.0) 
    star_rep.draw('test1')
    # exit()
    # [debug]
    # star_rep.draw_gamma()
    # star_rep.draw_normal_vector(filter=True)

    # init a graph manager
    graph_manager = GraphManager(star_rep)
    start_star_id = copy(graph_manager.start_id)
    current_star_id = copy(graph_manager.start_id)
    path, reach = graph_manager.find_path(current_star_id, goal_position)
    local_id = path['path_id'][-1]
    local_position = path['path'][-1]
   
    bias = np.array([0, 0])
    temp_position = np.array([1.0, 10.0])
    new_position = start_position + bias
    init_velocity = np.array([0.0, 0.0])
    
    reach_local = False
    reach_global = False
    update_hz = 40
    iter = 0
    interval = 10

    # judge if the robot is staying in the local position too long
    local_position_init = 0.1
    local_position_limit = copy(local_position_init)
    local_relax = False
    local_position_count = 0
    local_position_count_lim = 10

    # generate a color
    # color = np.random.rand(3,)
    safety_margin = 0.15
    local_position_limit = copy(safety_margin)

    while not reach_global and iter < 1000:
        # new_velocity, max_Gamma = modulation_velocity(new_position, local_position, graph_manager, safety_margin)
        # new_velocity = new_velocity / np.linalg.norm(new_velocity)
        new_velocity = modulation_round_robot(new_position, local_position, graph_manager, safety_margin)


        # if safety_max_Gamma <= 1.2 and safety_max_Gamma > 0.5:
        #     new_velocity = safety_velocity
        # # elif safety_max_Gamma < 1.05:
        # #     # merge new_velocity and safety_velocity with corresponding weight
        # new_velocity = new_velocity  + safety_velocity * 0

        # normalize the new_velocity
        last_new_position = copy(new_position)
        new_velocity = new_velocity / np.linalg.norm(new_velocity)
        new_position = new_position + new_velocity * 1/update_hz

        new_velocity = new_velocity * 0.1
        
        if iter % interval == 0:
            # plt.scatter(new_position[0], new_position[1], c='black', s=10)
            plt.arrow(new_position[0], new_position[1], new_velocity[0], new_velocity[1], head_width=0.1, head_length=0.1, fc='black', ec='black', animated=True)

        iter += 1

        # judge if reach the local position
        if np.linalg.norm(local_position - new_position) < local_position_limit and \
            np.linalg.norm(local_position - goal_position) > 0.001:
            print('reach the local position')
            reach_local = True
            # if local_relax:
            #     local_position_limit = copy(local_position_init)
            #     local_relax = False

        # judge if reach the global position
        if np.linalg.norm(goal_position - new_position) < 0.2:
            print('reach the global position')
            reach_global = True
            reach_local = False

        # if reach local, generate a new star node and find a new path
        if reach_local:
            print('test', local_position, new_position)
            # generate a new star node
            de_obs = laser.state2obs(new_position, robot_yaw, False)
            laser_points = de_obs['point']

            find_obstacle = False
            in_other_starshape = False
            # judge if there are points around the local goal
            for i in range(len(laser_points)):
                if np.linalg.norm(laser_points[i] - local_position) < 0.2:
                    print('remove the obstacle', np.linalg.norm(laser_points[i] - local_position))
                    find_obstacle = True
                    # remove the node in the graph
                    graph_manager.remove_node(local_id)
                    # generate a new path
                    path, reach = graph_manager.find_path(current_star_id, goal_position)
                    local_id = path['path_id'][-1]
                    local_position = path['path'][-1]
                   
                    break

            if not find_obstacle:
                print('extend obstacle')
                # laser_points = laser_points.tolist()
                star_rep = StarshapedRep((laser_points*resolusion), new_position, robot_radius=0.0)
                graph_manager.extend_star_node(local_id, star_rep, new_position)
                current_star_id = copy(local_id)
                star_rep.draw('test1')
                path, reach = graph_manager.find_path(current_star_id, goal_position)
                local_id = path['path_id'][-1]
                local_position = path['path'][-1]
            
            reach_local = False
            
        print(iter, 'local position:', local_position, 'pos', new_velocity, np.linalg.norm(last_new_position - new_position))
        # if staying in position too long, relax the local judge condition
        # if np.linalg.norm(last_new_position - new_position) < 0.1:
        #     local_position_count += 1
        #     if local_position_count > local_position_count_lim:
        #         local_relax = True
        #         local_position_limit += 0.1
        #         local_position_count = 0
        # else:
        #     local_position_count = 0

        if iter % interval == 0:
            plt.axis('equal')
            # plt.tight_layout()
            # plt.savefig('results/finding_path'+str(iter)+'.png', dpi=300)
    plt.tight_layout()
    plt.savefig('finding_path.png', dpi=300)
    # evaluation_modulation_result(xlim, ylim, graph_manager, goal_position)