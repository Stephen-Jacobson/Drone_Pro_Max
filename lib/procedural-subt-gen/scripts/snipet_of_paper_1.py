from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from random import randrange

tn = TunnelNetwork(initial_node=False)
n1 = Node(0, 0, randrange(-5, 5))
n2 = Node(25, 0, randrange(-5, 5))
n3 = Node(50, 0, randrange(-5, 5))
n4 = Node(70, 20, randrange(-5, 5))
n5 = Node(50, 40, randrange(-5, 5))
n6 = Node(25, 40, randrange(-5, 5))
n7 = Node(0, 40, randrange(-5, 5))
t1 = tn.add_tunnel(Tunnel(nodes=(n1, n2, n3, n4, n5, n6, n7)))
t2 = tn.add_tunnel(Tunnel(nodes=(n2, n6)))
t3 = tn.add_tunnel(Tunnel(nodes=(n3, n5)))
tnmg = TunnelNetworkMeshGenerator(tn)
tnmg.compute_all()

import pyvista as pv
from pyvista.plotting.plotting import Plotter
import numpy as np
from subt_proc_gen.display_functions import plot_spline, plot_nodes

plotter = Plotter(off_screen=True)
plotter.set_background("w")
plotter.add_mesh(tnmg.pyvista_mesh, style="wireframe")
plot_spline(plotter, t1.spline, color="purple", radius=0.5)
plot_spline(plotter, t2.spline, color="orange", radius=0.5)
plot_spline(plotter, t3.spline, color="yellow", radius=0.5)
plot_nodes(plotter, tn.nodes, radius=1, color="k")
plotter.add_lines(np.array(((0, 0, 0), (100, 0, 0))), color="r")
plotter.add_lines(np.array(((0, 0, 0), (0, 100, 0))), color="g")
plotter.add_lines(np.array(((0, 0, 0), (0, 0, 100))), color="b")
plotter.camera_position = [
    (-33.596806072562565, -58.05020103620892, 55.82768631888251),
    (28.83608842117635, 22.63549910178521, -4.012262037559747),
    (0.48932697416170395, 0.24262198828776566, 0.8376715843079807),
]
import os

images_path = "/home/lorenzo/images/papers/subt_proc_gen"
os.makedirs(images_path, exist_ok=True)
plotter.show(screenshot=os.path.join(images_path, "tn_manual_nodes_1.png"))
print(plotter.camera_position)
