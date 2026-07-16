from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
import os
import pyvista as pv
from pyvista.plotting.plotting import Plotter
import numpy as np
from subt_proc_gen.display_functions import plot_spline, plot_nodes, plot_graph, plot_splines

for i in range(20):
    while True:
        tn = TunnelNetwork(initial_node=False)
        n1 = Node(0, 0, 0)
        while True:
            t1 = Tunnel.grown(i_node=n1)
            if tn.add_tunnel_if_no_collision(t1):
                break
        while True:
            t2 = Tunnel.grown(i_node=tn.get_random_node())
            if tn.add_tunnel_if_no_collision(t2):
                break
        while True:
            ni, nf = tn.get_n_random_nodes(2)
            t3 = Tunnel.connector(inode=ni, fnode=nf)
            if tn.add_tunnel_if_no_collision(t3):
                break
        plotter = Plotter()
        plot_splines(plotter, tn)
        plot_graph(plotter, tn)
        plotter.show()
        decission = input("Mesh this graph?")
        if "y" in decission.lower():
            tnmg = TunnelNetworkMeshGenerator(tn)
            tnmg.compute_all()
            break

    plotter = Plotter(off_screen=False)
    # plotter = Plotter()
    plotter.set_background("w")
    plotter.add_mesh(tnmg.pyvista_mesh, style="wireframe")
    plot_spline(plotter, t1.spline, color="purple", radius=0.5)
    plot_spline(plotter, t2.spline, color="orange", radius=0.5)
    plot_spline(plotter, t3.spline, color="yellow", radius=0.5)
    plot_nodes(plotter, tn.nodes, radius=1, color="k")
    plotter.add_lines(np.array(((0, 0, 0), (100, 0, 0))), color="r")
    plotter.add_lines(np.array(((0, 0, 0), (0, 100, 0))), color="g")
    plotter.add_lines(np.array(((0, 0, 0), (0, 0, 100))), color="b")
    plotter.show()
    camera_position = plotter.camera_position
    plotter = Plotter(off_screen=True)
    # plotter = Plotter()
    plotter.set_background("w")
    plotter.add_mesh(tnmg.pyvista_mesh, style="wireframe")
    plot_spline(plotter, t1.spline, color="purple", radius=0.5)
    plot_spline(plotter, t2.spline, color="orange", radius=0.5)
    plot_spline(plotter, t3.spline, color="yellow", radius=0.5)
    plot_nodes(plotter, tn.nodes, radius=1, color="k")
    plotter.add_lines(np.array(((0, 0, 0), (100, 0, 0))), color="r")
    plotter.add_lines(np.array(((0, 0, 0), (0, 100, 0))), color="g")
    plotter.add_lines(np.array(((0, 0, 0), (0, 0, 100))), color="b")
    plotter.camera_position = camera_position
    images_path = "/home/lorenzo/images/papers/subt_proc_gen"
    os.makedirs(images_path, exist_ok=True)
    plotter.show(screenshot=os.path.join(images_path, f"tn_procedural_{i}.png"))
    # plotter.show()
    # print(plotter.camera_position)
