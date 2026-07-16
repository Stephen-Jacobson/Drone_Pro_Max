from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
import os
import pyvista as pv
from pyvista.plotting.plotting import Plotter
import numpy as np
from subt_proc_gen.display_functions import plot_spline, plot_nodes, plot_graph, plot_splines, plot_xyz_axis


img_save_path = "/home/lorenzo/Documents/conferences/IROS_2024/poster/arbitrarily_complex"


class storage:
    def __init__(self, data):
        self.data = data


n_saved = storage(0)


def do_the_plotting(plotter, tnmg):
    plotter.set_background("w")
    plotter.add_mesh(tnmg.pyvista_mesh, style="wireframe")
    plot_splines(plotter, tnmg._tunnel_network, color="red", radius=0.5)
    plot_nodes(plotter, tnmg._tunnel_network.nodes, radius=1, color="k")


def plot_and_show(tnmg: TunnelNetworkMeshGenerator):
    plotter = Plotter(off_screen=False, window_size=(2000, 1000))
    do_the_plotting(plotter, tnmg)
    plotter.show()
    camera_position = plotter.camera_position
    plotter = Plotter(off_screen=True, window_size=(2000, 1000))
    do_the_plotting(plotter, tnmg)
    plotter.camera_position = camera_position
    file_path = os.path.join(img_save_path, f"{n_saved.data:03d}.png")
    plotter.show(screenshot=file_path)
    n_saved.data += 1


def main():
    for n_grown, n_connector in ((4, 2), (8, 4), (12, 6), (20, 10)):
        tn = TunnelNetwork(initial_node=False)
        print(f"\nn_grown: {n_grown}")
        print(f"n_connector: {n_connector}")
        for _ in range(n_grown):
            tn.add_random_grown_tunnel(n_trials=100)
        while len(tn.tunnels) < n_grown + n_connector:
            tn.add_random_connector_tunnel(n_trials=10)
        tnmg = TunnelNetworkMeshGenerator(tn)
        tnmg.compute_all()
        plot_and_show(tnmg)


if __name__ == "__main__":
    main()
