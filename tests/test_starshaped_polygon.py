from obstacles import Frame,  StarshapedPolygon
from dynamic_obstacle_avoidance.obstacles import Polygon
from utils import generate_convex_polygon, draw_shapely_polygon, generate_star_polygon
from starshaped_hull import cluster_and_starify, draw_clustering, draw_adm_ker, draw_star_hull
import numpy as np
import matplotlib.pyplot as plt
from vartools.dynamical_systems import LinearSystem
from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

visualize = True

def test_starshaped_obstacle():

    avg_radius = 2
    xlim = [-2*avg_radius, 2*avg_radius]
    ylim = xlim
    pol = StarshapedPolygon(generate_star_polygon([0, 0], avg_radius, irregularity=0.1, spikiness=0.5, num_vertices=10))

    print(pol.distance_function(np.array([0, 0])))

    while True:
        x = np.array([np.random.uniform(*xlim), np.random.uniform(*ylim)])
        if pol.exterior_point(x):
            break
    b = pol.boundary_mapping(x)
    n = pol.normal(x)
    tp = pol.tangent_points(x)
    dir = pol.reference_direction(x)

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

    for i in np.linspace(0, 2 * np.pi, 100):
        x = pol.xr() + 100*np.array([np.cos(i), np.sin(i)])
        b = pol.boundary_mapping(x)
        b_list.append(b)
        n = pol.normal(b)
        ax.quiver(*b, *n)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    print("es")
    plt.savefig("starshaped_polygon.png", dpi=300)

    plt.cla()
    margin_absolut = 0.05
    center_position = np.array([0.0, 0.0])
    b_list = np.array(b_list).T

    obstacle_environment = []
    obstacle_environment.append(
        # StarshapedFlower(
        #     center_position=np.array([0, 0]),
        #     is_boundary=True,
        # )
        Polygon(
                edge_points=b_list,
                is_boundary=True,
                tail_effect=False,
            )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.2, 0.0]),
        maximum_velocity=0.5,
        distance_decrease=0.3,
    )

    if visualize:
        plt.close("all")

        # x_lim = [-10, 10]
        # y_lim = [-10, 10]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_range=xlim,
            y_range=ylim,
            point_grid=100,
            obstacle_list=obstacle_environment,
            pos_attractor=initial_dynamics.attractor_position,
            dynamical_system=initial_dynamics.evaluate,
            noTicks=True,
            automatic_reference_point=False,
            show_streamplot=True,
            draw_vectorField=True,
            normalize_vectors=False,
        )

        plt.grid()
        plt.savefig("simple_vectorfield_inside.png", dpi=300)

    # plt.show()

    for i in range(10):
        point = b_list[:, i]
        

test_starshaped_obstacle()