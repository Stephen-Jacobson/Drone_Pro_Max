import os
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import *
from subt_proc_gen.PARAMS import N_ANGLES_PER_CIRCLE
from subt_proc_gen.mesh_generation import TunnelWithMesh, TunnelPtClGenParams
import matplotlib
import pickle

matplotlib.rcParams.update({"font.size": 25})

interactive = False
save_file_name = "intersection_cleaning_4"
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
plot_axis_related_things = True
plot_pointcloud_without_noise = True
noise = 0.0
flatten_floor = True
size_of_points = 30
t1_pts_to_del_color = "y"
t2_pts_to_del_color = "y"

show = [True, True, False]


def main():
    save_path_1 = os.path.join(save_folder, save_file_name + "_1.png")
    save_path_2 = os.path.join(save_folder, save_file_name + "_2.png")
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
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
    tunnel_2 = Tunnel(graph, tunnel_params)
    tunnel_2.compute(tunnel_1.nodes[2])
    axis_points_1 = tunnel_1.spline.discretize(MIN_DIST_OF_MESH_POINTS)[1]
    axis_points_2 = tunnel_2.spline.discretize(MIN_DIST_OF_MESH_POINTS)[1]

    ####################################################################################################################################
    # 	PLOT THE AXIS-RELATED THINGS
    ####################################################################################################################################
    plotter = pv.Plotter(off_screen=not interactive)
    plotter.set_background("white")
    plotter.window_size = window_size
    if plot_axis_related_things:
        # Plot the arrows
        all_axis_arrows = [[], []]
        for n_axis, axis_points in enumerate([axis_points_1, axis_points_2]):
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
        # Create the spheres
        all_axis_spheres = [[], []]
        for n_axis, axis_points in enumerate([axis_points_1, axis_points_2]):
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
        twm2 = TunnelWithMesh(tunnel_2, meshing_params=params)
        pts1 = twm1.central_points
        pts2 = twm2.central_points
        ids1 = []
        ids2 = []
        for id1, pt1 in enumerate(pts1):
            if twm2.is_point_inside(pt1):
                ids1.append(id1)
        for id2, pt2 in enumerate(pts2):
            if twm1.is_point_inside(pt2):
                ids2.append(id2)
        pts1_nok = pts1[ids1]
        pts2_nok = pts2[ids2]
        pts1_ok = np.delete(pts1, ids1, axis=0)
        pts2_ok = np.delete(pts2, ids2, axis=0)
        pts1_ok = np.concatenate([pts1_ok, twm1._floor_points], axis=0)
        pts2_ok = np.concatenate([pts2_ok, twm2._floor_points], axis=0)
        pc1_ok = pv.PolyData(pts1_ok)
        pc2_ok = pv.PolyData(pts2_ok)
        pc1_nok = pv.PolyData(pts1_nok)
        pc2_nok = pv.PolyData(pts2_nok)
        if show[0]:
            plotter.add_mesh(
                pc1_ok,
                color="g",
                show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
        if show[1]:
            plotter.add_mesh(
                pc2_ok,
                color="m",
                show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
        if show[2]:
            plotter.add_mesh(
                pc1_nok,
                color=t1_pts_to_del_color,
                show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
        if show[3]:
            plotter.add_mesh(
                pc2_nok,
                color=t2_pts_to_del_color,
                show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=size_of_points * size_mult,
            )
    # CAMERA PARAMETERS
    cpos1 = (12.263109855335816, -15.439865265681345, 9.341490494085114)
    cfpt1 = (18.52788972550311, -4.1186584686806444, 0.08970480496248534)
    cr1 = 46.71755840988013
    cpos2 = (32.359539537453045, -7.064128555564271, 9.00584406349876)
    cfpt2 = (18.993666752563772, -3.984861227766646, 0.9507657881694531)
    cr2 = -80.07716059069989
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
        plotter.show(screenshot=save_path_1,
                     window_size=window_size, auto_close=False)
        plotter.camera.position = cpos2
        plotter.camera.focal_point = cfpt2
        plotter.camera.roll = cr2
        plotter.show(screenshot=save_path_2, window_size=window_size)


if __name__ == "__main__":
    main()
