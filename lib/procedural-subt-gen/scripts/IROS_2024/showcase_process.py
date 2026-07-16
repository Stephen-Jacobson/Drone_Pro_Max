from subt_proc_gen.tunnel import TunnelNetwork, Node, Tunnel
from subt_proc_gen.geometry import Vector3D
from subt_proc_gen.display_functions import (
    plot_graph,
    plot_tunnel_network_graph,
    plot_mesh,
    plot_splines,
    plot_tunnel_ptcls,
    plot_intersection_ptcls,
    plot_intersection_nodes,
)
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from pyvista import Plotter
import pyvista as pv
import numpy as np
import os

pv.rcParams["transparent_background"] = True
BASE_FOLDER = "/home/lorenzo/Documents/conferences/IROS_2024/poster"


def create_nodes():
    nodes = [None for _ in range(1 + 11)]
    nodes[1] = Node(0, 0, 0)
    nodes[2] = Node(50, 0, 0)
    nodes[3] = Node(50, 50, 0)
    nodes[4] = Node(50, 100, 0)
    nodes[5] = Node(100, 50, 0)
    nodes[6] = Node(100, 0, 0)
    nodes[7] = Node(50, -50, 0)
    nodes[8] = Node(100, -50, 0)
    nodes[9] = Node(150, -50, 0)
    nodes[10] = Node(150, 0, 0)
    nodes[11] = Node(200, 50, 0)
    return nodes


def add_noise_to_nodes(nodes):
    for node in nodes:
        if isinstance(node, Node):
            node.modify_pose(Vector3D(np.random.uniform(-10, 10, 3)))
    return nodes


def create_tunnels(nodes):
    tunnels = [None for _ in range(1 + 12)]

    tunnels[1] = Tunnel.connector(nodes[1], nodes[2])
    tunnels[2] = Tunnel.connector(nodes[2], nodes[6])
    tunnels[3] = Tunnel.connector(nodes[6], nodes[10])
    tunnels[4] = Tunnel.connector(nodes[2], nodes[3])
    tunnels[5] = Tunnel.connector(nodes[6], nodes[5])
    tunnels[6] = Tunnel.connector(nodes[10], nodes[11])
    tunnels[7] = Tunnel.connector(nodes[3], nodes[4])
    tunnels[8] = Tunnel.connector(nodes[3], nodes[5])
    tunnels[9] = Tunnel.connector(nodes[2], nodes[7])
    tunnels[10] = Tunnel.connector(nodes[6], nodes[9])
    tunnels[11] = Tunnel.connector(nodes[7], nodes[8])
    tunnels[12] = Tunnel.connector(nodes[8], nodes[9])
    del tunnels[0]
    return tunnels


def plot_graph_initial(tn: TunnelNetwork):
    plotter = Plotter()
    plot_tunnel_network_graph(plotter, tn)
    plotter.show()
    return plotter.camera_position


def plot_graph_screenshot(tn: TunnelNetwork, camera_pose, filename):
    plotter = Plotter(off_screen=True)
    plot_tunnel_network_graph(plotter, tn)
    plotter.camera_position = camera_pose
    path_to_file = os.path.join(BASE_FOLDER, filename)
    plotter.show(screenshot=path_to_file)


def plot_splines_screenshot(tn: TunnelNetwork, camera_pose, filename):
    plotter = Plotter(off_screen=True)
    plot_intersection_nodes(plotter, tn, color="k")
    plot_splines(plotter, tn, radius=0.5)
    plotter.camera_position = camera_pose
    path_to_file = os.path.join(BASE_FOLDER, filename)
    plotter.show(screenshot=path_to_file)


def plot_ptcl_screenshot(tnmg: TunnelNetworkMeshGenerator, camera_pose, filename):
    plotter = Plotter(off_screen=True)
    plot_splines(plotter, tnmg._tunnel_network, radius=0.5)
    plot_intersection_ptcls(plotter, tnmg, color="b", size=0.1)
    plot_tunnel_ptcls(plotter, tnmg, color="b", size=0.1)
    plotter.camera_position = camera_pose
    path_to_file = os.path.join(BASE_FOLDER, filename)
    plotter.show(screenshot=path_to_file)


def plot_mesh_screenshot(tnmg: TunnelNetworkMeshGenerator, camera_pose, filename):
    plotter = Plotter(off_screen=True)
    plot_mesh(plotter, tnmg)
    plotter.camera_position = camera_pose
    path_to_file = os.path.join(BASE_FOLDER, filename)
    plotter.show(screenshot=path_to_file)


def main():
    nodes = create_nodes()
    add_noise_to_nodes(nodes)
    tunnels = create_tunnels(nodes)
    tn = TunnelNetwork(initial_node=False)
    for tunnel in tunnels:
        tn.add_tunnel(tunnel)
    camera_position = plot_graph_initial(tn)
    plot_graph_screenshot(tn, camera_position, "0_graph.png")
    plot_splines_screenshot(tn, camera_position, "1_splines.png")
    tnmg = TunnelNetworkMeshGenerator(tn)
    tnmg.compute_all()
    plot_mesh_screenshot(tnmg, camera_position, "3_mesh.png")
    plot_ptcl_screenshot(tnmg, camera_position, "2_ptcl.png")


if __name__ == "__main__":
    main()
