import os
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import *
from subt_proc_gen.mesh_generation import TunnelWithMesh, TunnelPtClGenParams
import matplotlib
import pickle

matplotlib.rcParams.update({"font.size": 25})

interactive = False
save_file_name = "mesh_generation_4"
if interactive:
    window_size = 1000, 1000
    size_mult = 1
else:
    window_size = 3000, 3000
    size_mult = 3
save_folder = "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures"
arrow_color = ["r", "r"]
axis_point_color = ["b", "b"]
backgroudn_color = "white"
axis_point_radius = 0.1
plot_pointcloud_without_noise = True
noise = 0.3
flatten_floor = True
size_of_points = 30
first_ring_color = "c"
rest_of_the_points_color = "gray"
floor_points_color = "orange"
first_ring_separation_number = N_ANGLES_PER_CIRCLE - 19 * flatten_floor
show = [True, True, flatten_floor]


def main():
    save_path_1 = os.path.join(save_folder, save_file_name + "_1.png")
    save_path_2 = os.path.join(save_folder, save_file_name + "_2.png")
    # Generate the graph
    graph = TunnelNetwork()
    central_node = CaveNode()
    tunnel_params = TunnelParams(
        {
            "distance": 30,
            "starting_direction": angles_to_vector((np.deg2rad(-30), np.deg2rad(0))),
            "horizontal_tendency": np.deg2rad(40),
            "horizontal_noise": np.deg2rad(0),
            "vertical_tendency": np.deg2rad(0),
            "vertical_noise": np.deg2rad(0),
            "segment_length": 10,
            "segment_length_noise": 0,
            "node_position_noise": 0,
        }
    )
    tunnel_1 = Tunnel(graph, tunnel_params)
    tunnel_1.compute(central_node)
    tunnel_params["starting_direction"] = angles_to_vector(
        (np.deg2rad(-50), np.deg2rad(0))
    )
    axis_points_1 = tunnel_1.spline.discretize(MIN_DIST_OF_MESH_POINTS)[1]

    ####################################################################################################################################
    # 	PLOT THE AXIS-RELATED THINGS
    ####################################################################################################################################
    plotter = pv.Plotter(off_screen=not interactive)
    plotter.set_background("white")
    plotter.window_size = window_size
    # Plot the arrows
    for n_axis, axis_points in enumerate([axis_points_1]):
        for n_point in range(1, len(axis_points_1)):
            start = axis_points[n_point - 1]
            end = axis_points[n_point]
            direction = end - start
            arrow = pv.Arrow(
                start=start,
                direction=direction,
                tip_length=0.25 / MIN_DIST_OF_MESH_POINTS,
                tip_radius=0.1 / MIN_DIST_OF_MESH_POINTS,
                shaft_radius=0.05 / MIN_DIST_OF_MESH_POINTS,
                scale=MIN_DIST_OF_MESH_POINTS,
            )
            plotter.add_mesh(arrow, color=arrow_color[n_axis])
    # Plot Spheres
    for n_axis, axis_points in enumerate([axis_points_1]):
        for n_point, axis_point in enumerate(axis_points):
            if n_point == len(axis_points_1) - 1:
                continue
            sphere = pv.Sphere(
                radius=axis_point_radius,
                center=axis_point,
            )
            plotter.add_mesh(sphere, color=axis_point_color[n_axis])

    ####################################################################################################################################
    # 	Plot the pointcloud
    ####################################################################################################################################
    if plot_pointcloud_without_noise:
        params = TunnelPtClGenParams(
            {
                "roughness": noise,
                "flatten_floor": flatten_floor,
                "floor_to_axis_distance": 1,
                "radius": 2,
            }
        )
        twm1 = TunnelWithMesh(tunnel_1, meshing_params=params)
        pts1 = twm1.central_points
        first_ring_points = pts1[:first_ring_separation_number]
        rest_of_the_points = pts1[first_ring_separation_number:]
        floor_points = twm1._floor_points
        if show[0]:
            plotter.add_mesh(
                first_ring_points,
                color=first_ring_color,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
        if show[1]:
            plotter.add_mesh(
                rest_of_the_points,
                color=rest_of_the_points_color,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
        if show[2]:
            plotter.add_mesh(
                floor_points,
                color=floor_points_color,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )

    # CAMERA PARAMETERS
    cpos1 = (-15.945418823045705, -1.0636138978704643, 13.898379345134758)
    cfpt1 = (13.035282192024912, -0.41908787282745563, -4.054063608336675)
    cr1 = 79.4557275207785
    cpos2 = (-11.843470490563574, 6.59792186315452, 7.735660994695985)
    cfpt2 = (11.922466606554108, -3.59260552114094, -3.4623617162215345)
    cr2 = 99.23827287385221
    _ = plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=1,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )

    if interactive:
        plotter.camera.position = cpos1
        plotter.camera.focal_point = cfpt1
        plotter.camera.roll = cr1
        plotter.show()
        print("cpos =", plotter.camera.position)
        print("cfpt=", plotter.camera.focal_point)
        print("cr =", plotter.camera.roll)
    else:
        plotter.camera.position = cpos1
        plotter.camera.focal_point = cfpt1
        plotter.camera.roll = cr1
        plotter.show(screenshot=save_path_1, window_size=window_size, auto_close=False)
        plotter.camera.position = cpos2
        plotter.camera.focal_point = cfpt2
        plotter.camera.roll = cr2
        plotter.show(screenshot=save_path_2, window_size=window_size)


if __name__ == "__main__":
    main()
