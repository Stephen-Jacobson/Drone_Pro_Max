import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import debug_plot, plot_spline_2d
from subt_proc_gen.tunnel import *
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh, TunnelPtClGenParams
import random
from subt_proc_gen.helper_functions import what_points_are_close
import matplotlib

matplotlib.rcParams.update({"font.size": 25})

show_grow_tunnel = False
spline_color = "b"
distance_between_spline_points = 7
color_of_close_points = "r"
color_of_axis_points = "gray"
final_spline_color = "g"
spline_zorder = -3
angle_of_candidate_tunnel = 70
plt_axis_lenght = 205
min_x = -7
min_y = -plt_axis_lenght / 2 - 25
name = "collision_3"


def main():
    fig = plt.figure(figsize=(10, 10))
    for i in range(1):
        for i in range(1):
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            # Generate the graph
            graph = TunnelNetwork()
            central_node = CaveNode()
            tunnel_params = TunnelParams(
                {
                    "distance": 150,
                    "starting_direction": angles_to_vector(
                        (np.deg2rad(-30), np.deg2rad(-5))
                    ),
                    "horizontal_tendency": np.deg2rad(-5),
                    "horizontal_noise": np.deg2rad(0),
                    "vertical_tendency": np.deg2rad(-5),
                    "vertical_noise": np.deg2rad(0),
                    "segment_length": 30,
                    "segment_length_noise": 0,
                    "node_position_noise": 0,
                }
            )
            node2 = CaveNode(np.array((0, 10, 0)))
            node3 = CaveNode(np.array((10, 10, 0)))
            node4 = CaveNode(np.array((10, 0, 0)))
            t1 = Tunnel(graph, params=tunnel_params)
            t1.set_nodes([central_node, node2, node3, node4, central_node])
            tnwm = TunnelNetworkWithMesh(
                graph, meshing_params=TunnelPtClGenParams())
            plotter = pv.Plotter()
            ptcl = pv.PolyData(tnwm.mesh_points_and_normals()[0])
            plotter.add_mesh(ptcl)
            plotter.show()
            exit()
            t1.compute(initial_node=central_node)
            tunnel_params["starting_direction"] = angles_to_vector(
                (np.deg2rad(20), np.deg2rad(-5))
            )
            t2 = Tunnel(graph, params=tunnel_params)
            t2.compute(initial_node=central_node)
            tunnel_params["distance"] = 110
            tunnel_params["starting_direction"] = angles_to_vector(
                (np.deg2rad(0), np.deg2rad(-5))
            )
            tunnel_params["segment_length"] = 20
            t3 = Tunnel(graph, params=tunnel_params)
            t3.compute(initial_node=central_node, do_checks=False)
            tunnel_params["starting_direction"] = angles_to_vector(
                (np.deg2rad(angle_of_candidate_tunnel), np.deg2rad(0))
            )
            t_show = Tunnel(
                graph,
                params=tunnel_params,
            )
            if show_grow_tunnel:
                t_show.compute(
                    initial_node=t1.nodes[-2],
                    do_checks=False,
                )
            else:
                t_show.compute(
                    initial_node=t1.nodes[-1],
                    final_node=t2.nodes[-1],
                    do_checks=False,
                )
            # Plot the splines
            for n_tunnel, tunnel in enumerate(graph.tunnels):
                if n_tunnel == len(graph.tunnels) - 1:
                    plot_spline_2d(
                        tunnel.spline, color=final_spline_color, zorder=spline_zorder
                    )
                else:
                    plot_spline_2d(
                        tunnel.spline, color=spline_color, zorder=spline_zorder
                    )
            # Plot the nodes of the graph
            for node in graph.nodes:
                x, y, z = node.xyz
                plt.scatter(
                    x=x,
                    y=y,
                    s=500,
                    zorder=-1,
                    edgecolors="k",
                    facecolors="none",
                    linewidths=3,
                )
            p1 = t1.spline.discretize(distance_between_spline_points)[1]
            plt.scatter(
                x=p1[:, 0], y=p1[:, 1], alpha=1, s=100, c=color_of_axis_points, zorder=0
            )
            p2 = t2.spline.discretize(distance_between_spline_points)[1]
            plt.scatter(
                x=p2[:, 0], y=p2[:, 1], alpha=1, s=100, c=color_of_axis_points, zorder=0
            )
            p3 = t3.spline.discretize(distance_between_spline_points)[1]
            plt.scatter(
                x=p3[:, 0], y=p3[:, 1], alpha=1, s=100, c=color_of_axis_points, zorder=0
            )
            p_show = t_show.relevant_points_for_collision(
                distance_between_points=distance_between_spline_points,
                distance_to_intersection=MIN_DIST_OF_TUNNEL_COLLISSIONS * 1.5,
            )
            plt.scatter(
                x=p_show[:, 0],
                y=p_show[:, 1],
                alpha=1,
                s=100,
                c=color_of_axis_points,
                zorder=0,
            )
            ids3, ids_show = what_points_are_close(p3, p_show, min_dist=10)
            plt.scatter(
                x=p_show[ids_show, 0],
                y=p_show[ids_show, 1],
                alpha=1,
                s=100,
                c=color_of_close_points,
                zorder=0,
            )
            plt.scatter(
                x=p3[ids3, 0],
                y=p3[ids3, 1],
                alpha=1,
                s=100,
                c=color_of_close_points,
                zorder=0,
            )

            plt.gca().set_xlim(min_x, min_x + plt_axis_lenght)
            plt.gca().set_ylim(min_y, min_y + plt_axis_lenght)
            # plt.show()
            plt.draw()
            if True:
                plt.savefig(
                    "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures/{}.svg".format(
                        name
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )
            if False:
                with open("datafiles/graph.pkl", "wb") as f:
                    pickle.dump(graph, f)


if __name__ == "__main__":
    main()
