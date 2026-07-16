import numpy as np
import matplotlib.pyplot as plt

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.tunnel import *
import random
import pyvista as pv
import pickle

save_graph = True
use_saved_graph = True
n_grown_tunnels = 1
n_connecting_tunnels = 0
n_environments = 1
path_to_screenshot = (
    "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures/mesh_3.png"
)


def main():
    for n_network in range(n_environments):
        # Generate the graph
        fig = plt.figure(figsize=(10, 10))
        graph = TunnelNetwork()
        central_node = CaveNode()
        for i in range(n_grown_tunnels):
            print(f"Grown tunnel {i+1} out of {n_grown_tunnels}")
            tunnel_params = TunnelParams()
            tunnel_params.random()
            if i == 0:
                tunnel = Tunnel(graph, params=tunnel_params)
                success = tunnel.compute(
                    initial_node=central_node,
                )
                continue
            success = False
            while not success:
                node = random.choice(graph.nodes)
                if len(node.connected_nodes) < 2:
                    continue
                tunnel_params.random()
                tunnel = Tunnel(graph, params=tunnel_params)
                success = tunnel.compute(initial_node=node)
        for i in range(n_connecting_tunnels):
            print(f"Connecting tunnel {i+1} out of {n_connecting_tunnels}")
            success = False
            while not success:
                i_node = random.choice(graph.nodes)
                f_node = random.choice(graph.nodes)
                assert isinstance(i_node, CaveNode)
                assert isinstance(f_node, CaveNode)
                if i_node is f_node:
                    continue
                same_tunnel = False
                for tunnel in i_node.tunnels:
                    if tunnel in f_node.tunnels:
                        same_tunnel = True
                        break
                if same_tunnel:
                    continue
                tunnel_params.random()
                last_tunnel = Tunnel(
                    graph,
                    params=tunnel_params,
                )
                success = last_tunnel.compute(
                    initial_node=i_node,
                    final_node=f_node,
                )
        if save_graph:
            with open("datafiles/graph.pkl", "wb+") as f:
                pickle.dump(graph, f)
        if use_saved_graph:
            with open("datafiles/graph.pkl", "rb") as f:
                graph = pickle.load(f)
        tunnel_network = TunnelNetworkWithMesh(graph, i_meshing_params="random")
        tunnel_network.clean_intersections()
        mesh, _ = tunnel_network.mesh()
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        f = np.c_[np.full(len(f), 3), f]
        pvmesh = pv.PolyData(v, f)
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.add_mesh(pvmesh, edge_color="black", color="tan", show_edges=True)
        plotter.show(screenshot=path_to_screenshot)
        cpos = plotter.camera.position
        cfpt = plotter.camera.focal_point
        cr = plotter.camera.roll
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(pvmesh, edge_color="black", color="tan", show_edges=True)
        plotter.camera.position = cpos
        plotter.camera.focal_point = cfpt
        plotter.camera.roll = cr
        plotter.show(screenshot=path_to_screenshot)


if __name__ == "__main__":
    main()
