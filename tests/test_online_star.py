
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.animation import FuncAnimation
from envs.gridmap import OccupancyGridMap
from planner.path_generator.astar import a_star
from planner.trajectory_generator.spline_interpolate import *
from starshaped_hull.graph import GraphManager

from sensors.laser_anyshape import Laser
from sklearn.cluster import DBSCAN
from test_starshaped_polygon import star_ds, starshaped_polygon
from starshaped_hull.starshaped_fit import StarshapedRep
from typing import Optional

import warnings
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import numpy.typing as npt

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


def compute_weight(Gamma, dist_limit=1.0, weight_pow=1.0):
    distances = np.array(Gamma)
    critical_points = distances >= dist_limit

    if np.sum(critical_points):
        if np.sum(critical_points) == 1:
            w =  critical_points * 1.0
            return w
        else:
            w = critical_points * 1.0 / np.sum(critical_points)
            return w
    
    distances = distances - dist_limit
    w = (1 / distances) ** weight_pow
    if np.sum(w) == 0:
        return w
    w = w / np.sum(w)

    return w


def compute_diagonal_matrix(
    Gamma,
    dim,
    rho=1,
    repulsion_coeff=1.0,
    tangent_eigenvalue_isometric=True,
    tangent_power=5,
    treat_obstacle_special=True,
    self_priority=1,
):
    print('Gamma ', Gamma)

    """Compute diagonal Matrix"""
    if Gamma <= 1 and treat_obstacle_special:
        # Point inside the obstacle
        delta_eigenvalue = 1
    else:
        delta_eigenvalue = 1.0 / abs(Gamma) ** (self_priority / rho)
    eigenvalue_reference = 1 - delta_eigenvalue * repulsion_coeff

    if tangent_eigenvalue_isometric:
        eigenvalue_tangent = 1 + delta_eigenvalue
    else:
        # Decreasing velocity in order to reach zero on surface
        eigenvalue_tangent = 1 - 1.0 / abs(Gamma) ** tangent_power
    return np.diag(
        np.hstack((eigenvalue_reference, np.ones(dim - 1) * eigenvalue_tangent))
    )


# @lru_cache(maxsize=10)
# TODO: expand cache for this [numpy-arrays]
# TODO: OR make cython
def get_orthogonal_basis(vector: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Get Orthonormal basis matrxi for an dimensional input vector."""
    # warnings.warn("Basis implementation is not continuous.") (?! problem?)
    if not (vector_norm := np.linalg.norm(vector)):
        warnings.warn("Zero norm-vector.")
        return np.eye(vector.shape[0])

    vector = vector / vector_norm

    dim = vector.shape[0]
    if dim <= 1:
        return vector.reshape((dim, dim))

    basis_matrix = np.zeros((dim, dim))

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-basis_matrix[1, 0], basis_matrix[0, 0]])
    else:
        # exit
        warnings.warn("Basis dim>2 implementation is not continuous.")
        exit()

    return basis_matrix


class UnitDirectionError(Exception):
    def __init__(self, message="Error with Unit Direction Handling"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


def get_directional_weighted_sum(
    null_direction: np.ndarray,
    weights: npt.ArrayLike,
    directions: np.ndarray,
    unit_directions: list[np.ndarray] = None,
    total_weight: float = 1,
    normalize: bool = True,
    normalize_reference: bool = True,
) -> np.ndarray:
    """Weighted directional mean for inputs vector ]-pi, pi[ with respect to the null_direction

    Parameters
    ----------
    null_direction: basis direction for the angle-frame
    directions: the directions which the weighted sum is taken from
    unit_direction: list of unit direction
    weights: used for weighted sum
    total_weight: [<=1]
    normalize: variable of type Bool to decide if variables should be normalized

    Return
    ------
    summed_velocity: The weighted sum transformed back to the initial space
    """
    # TODO: this can be vastly speed up by removing the 'unit directions'
    weights = np.array(weights)

    ind_nonzero = np.logical_and(
        weights > 0, LA.norm(directions, axis=0)
    )  # non-negative

    null_direction = np.copy(null_direction)
    directions = directions[:, ind_nonzero]
    weights = weights[ind_nonzero]

    if total_weight > 1:
        weights = weights / np.sum(weights) * total_weight

    n_directions = weights.shape[0]
    if (n_directions == 1) and np.sum(weights) >= 1:
        return directions[:, 0] / LA.norm(directions[:, 0])

    dim = np.array(null_direction).shape[0]

    base = get_orthogonal_basis(vector=null_direction)
    if unit_directions is None:
        unit_directions = [
            UnitDirection(base).from_vector(directions[:, ii])
            for ii in range(directions.shape[1])
        ]
    else:
        for u_dir in unit_directions:
            u_dir.transform_to_base(base)

    summed_dir = UnitDirection(base).from_angle(np.zeros(dim - 1))
    for ii, u_dir in enumerate(unit_directions):
        summed_dir = summed_dir + u_dir * weights[ii]

    if True:
        return summed_dir.as_vector()

def compute_decomposition_matrix(obs, x_t):
    """Compute decomposition matrix and orthogonal matrix to basis"""
    normal_vector = obs._star_rep.get_normal_direction(x_t)
    reference_direction = obs._star_rep.get_reference_direction(x_t)
    
    # dot_prod = np.dot(normal_vector, reference_direction)

    E_orth = get_orthogonal_basis(normal_vector, normalize=True)
    E = np.copy((E_orth))
    # plot the reference direction, orthogonal vector and normal vector
    # plt.arrow(x_t[0], x_t[1], reference_direction[0], reference_direction[1], head_width=0.1, head_length=0.1, fc='b', ec='b')
    # plt.arrow(x_t[0], x_t[1], E_orth[0, 1], E_orth[1, 1], head_width=0.2, head_length=0.1, fc='y', ec='y')
    # plt.arrow(x_t[0], x_t[1], normal_vector[0], normal_vector[1], head_width=0.1, head_length=0.1, fc='g', ec='g')

    # [debug reference direction]
    print('ref', x_t, reference_direction, normal_vector)
    E[:, 0] = -reference_direction

    return E, E_orth

def get_relative_obstacle_velocity(
    position: np.ndarray,
    obstacle_list,
    E_orth: np.ndarray,
    weights: list,
    ind_obstacles: Optional[int] = None,
    gamma_list: Optional[list] = None,
    cut_off_gamma: float = 1e4,
    velocity_only_in_positive_normal_direction: bool = True,
    normal_weight_factor: float = 1.3,
) -> np.ndarray:
    """Get the relative obstacle velocity

    Parameters
    ----------
    E_orth: array which contains orthogonal matrix with repsect to the normal
    direction at <position>
    array of (dimension, dimensions, n_obstacles
    obstacle_list: list or <obstacle-conainter> with obstacles
    ind_obstacles: Inidicates which obstaces will be considered (array-like of int)
    gamma_list: Precalculated gamma-values (list of float) -
                It is adviced to use 'proportional' gamma values, rather
                than relative ones

    Return
    ------
    relative_velocity: array-like of float
    """
    n_obstacles = len(obstacle_list)

    if gamma_list is None:
        gamma_list = np.zeros(n_obstacles)
        for n in range(n_obstacles):
            gamma_list[n] = obs[n].get_gamma(position, in_global_frame=True)

    if ind_obstacles is None:
        ind_obstacles = gamma_list < cut_off_gamma
        gamma_list = gamma_list[ind_obstacles]

    obs = obstacle_list
    ind_obs = ind_obstacles
    dim = position.shape[0]

    xd_obs = np.zeros((dim))
    for ii, it_obs in zip(range(np.sum(ind_obs)), np.arange(n_obstacles)[ind_obs]):
        xd_obs_n = np.zeros(dim)

        # The Exponential term is very helpful as it help to avoid
        # the crazy rotation of the robot due to the rotation of the object
        if obs[it_obs].is_deforming:
            weight_deform = np.exp(-1 / 1 * (np.max([gamma_list[ii], 1]) - 1))
            vel_deformation = obs[it_obs].get_deformation_velocity(pos_relative[:, ii])

            if velocity_only_in_positive_normal_direction:
                vel_deformation_local = E_orth[:, :, ii].T.dot(vel_deformation)
                if (vel_deformation_local[0] > 0 and not obs[it_obs].is_boundary) or (
                    vel_deformation_local[0] < 0 and obs[it_obs].is_boundary
                ):
                    vel_deformation = np.zeros(vel_deformation.shape[0])

                else:
                    vel_deformation = E_orth[:, 0, ii].dot(vel_deformation_local[0])

            xd_obs_n += weight_deform * vel_deformation
        xd_obs = xd_obs + xd_obs_n * weights[ii]
    return xd_obs


def compute_modulation_matrix(D, E, weight, initial_velocity):
    # M(x) = E(x)*D(x)*E^(-1)
    # M = np.zeros((2, 2, len(weight)))
    vels = np.zeros((2, len(weight)))

    for i in range(len(weight)):
        # M[:, :, i] = E_orth[:, :, i].dot(D[:, :, i]).dot(np.linalg.inv(E_orth[:, :, i]))
        # if the first term of D is 1
        if D[0, 0, i] == 1:
            relative_vel = initial_velocity
        else:
            print('inv matrix', np.linalg.inv(E[:,:,i]))
            velocity_temp = np.linalg.pinv(E[:,:,i]).dot(initial_velocity)
            velocity_temp = D[:,:,i].dot(velocity_temp)
            relative_vel = E[:,:,i].dot(velocity_temp)

        vels[:, i] = relative_vel

    vel = np.zeros(2)
    for i in range(len(weight)):
        vel += np.array(vels[:, i]).T * weight[i]

    return vel.T

def modulation_velocity(position, goal_position, graph_manager):
    attractive_dir = goal_position - position
    initial_velocity = attractive_dir / np.linalg.norm(attractive_dir)
    print('init velocity', initial_velocity)

    obs_number = len(graph_manager._star_id_list)

    position = np.array(position)
    Gamma = np.zeros(len(graph_manager._star_id_list))
    for i in range(len(graph_manager._star_id_list)):
        id = graph_manager._star_id_list[i]
        Gamma[i] = graph_manager._nodes[id].get_gamma(position)
        # [debug GAMMA]
        # print('gamma', Gamma[i])
    weight = compute_weight(Gamma)
    # [debug weight]
    # print(weight, obs_number)

    E = np.zeros((2, 2, obs_number))
    D = np.zeros((2, 2, obs_number))
    E_orth = np.zeros((2, 2, obs_number))

    for n in range(obs_number):
        D[:, :, n] = compute_diagonal_matrix(Gamma[n], 2)
        E[:, :, n], E_orth[:, :, n] = compute_decomposition_matrix(graph_manager._nodes[graph_manager._star_id_list[n]], position)

    star_obs_list = []
    for i in range(len(graph_manager._star_id_list)):
        star_obs_list.append(graph_manager._nodes[graph_manager._star_id_list[i]]._star_rep)

    print('D', D)
    print('E', E)
    print('E_orth', E_orth)
    new_velocity = compute_modulation_matrix(D, E, weight, initial_velocity)
    # print(modulation_matrix)
    # new_velocity = modulation_matrix.dot(initial_velocity)

    return new_velocity
    

if __name__ == '__main__':
    map_file = 'maps/obstacle_map_occupancy.png'
    resolusion = 0.1    
    xlim = [-5, 20]
    ylim = [-5, 20]

    
    gridmap = OccupancyGridMap.from_png(map_file, resolusion)
    pM = PathManager(gridmap)
    # start_position = np.array([8.0, 5.5])
    start_position = np.array([4.0, 5.0])
    goal_position = np.array([8, 5.0])

    robot_c = start_position
    robot_yaw = 0.0
    
    # instance a laser class
    laser = Laser(beams=128)
    laser.set_map(gridmap)
    

    ## plot the map
    # pM.plot_map()
    ## plot laser points
    # plt.scatter(laser_points[:, 0], laser_points[:, 1], s=1, c='red', label='laser points')
    ## plot start position
    # plt.plot(start_position[0]/0.1, start_position[1]/0.1, 'go', label='start position')
    # plt.savefig('test_laser_points.png', dpi=300)
    
    # points = []
    # for i in range(len(laser_points)):
    #     points.append(tuple(laser_points[i]))

    plt.cla()

    # generate the laser points
    de_obs = laser.state2obs(robot_c, robot_yaw, False)
    laser_points = de_obs['point']
    # convert laser points' type to list 
    laser_points = laser_points.tolist()

    b_list = starshaped_polygon(laser_points, plot=False)
    # star_ds(b_list*resolusion, start_position, start_position, x_lim=xlim, y_lim=ylim, plot=False)
    # star_ds(laser_points, start_position)
    # plt.cla()
    star_node = StarshapedRep(b_list*resolusion, start_position)

    fig = plt.figure()
    star_node.draw('test1')
    # [debug]
    # star_node.draw_gamma()
    # star_node.draw_normal_vector(filter=True)


    graph_manager = GraphManager(star_node)
    path, reach = graph_manager.find_path(1, goal_position)
    id = path['path_id'][-1]
    local_position = path['path'][-1]
   
    # modulation_velocity(start_position, goal_position, graph_manager)
    # bias = np.array([0.9284,0.3713])
    bias = np.array([0, 0])
    temp_position = np.array([3.5, 8.0])
    new_position = temp_position + bias
    init_velocity = np.array([0.0, 0.0])
    
    # while not reach:
    #     new_position = path['path'][-1]
    #     de_obs = laser.state2obs(new_position, robot_yaw, False)
    #     laser_points = de_obs['point']
    #     laser_points = laser_points.tolist()
    #     b_list = starshaped_polygon(laser_points, plot=False)
    #     star_node = StarshapedRep(b_list*resolusion, new_position)
    #     plot_name = 'test' + str(plot_number)
    #     plot_number += 1
    #     star_node.draw(plot_name)
    #     graph_manager.extend_node(id, star_node)
    #     path, reach = graph_manager.find_path(1, goal_position)
    #     id = path['path_id'][-1]
    #     print(path, reach)

    reach_local = False
    update_hz = 10
    iter = 0
    interval = 2

    # generate a color
    color = np.random.rand(3,)

    while not reach_local and iter < 100:
        new_velocity = modulation_velocity(new_position, local_position, graph_manager)
        modulated_position = new_velocity + new_position
        # plot the arrow of the new velocity
        print('local position', local_position)
        attractive_dir = local_position - new_position
        init_velocity = attractive_dir / np.linalg.norm(attractive_dir)
        print('velocity', init_velocity, new_velocity, modulated_position)

        # normalize the new_velocity
        new_velocity = new_velocity / np.linalg.norm(new_velocity)
        new_position = new_position + new_velocity * 1/update_hz
        
        color = np.random.rand(3,)
        if iter % interval == 0:
            plt.arrow(new_position[0], new_position[1], new_velocity[0], new_velocity[1], head_width=0.1, head_length=0.1, fc=color*iter/100, ec=color*iter/100, animated=True)

        iter += 1

        # judge if reach the local position
        if np.linalg.norm(local_position - new_position) < 0.1:
            reach_local = True

        if iter % interval == 0:
            plt.savefig('finding_path.png', dpi=300)

    




