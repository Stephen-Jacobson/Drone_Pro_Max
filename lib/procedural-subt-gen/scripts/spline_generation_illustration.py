from subt_proc_gen.geometry import Spline3D, Point3D
from subt_proc_gen.display_functions import plot_spline, plot_graph, plot_nodes
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork, Node
from pyvista.plotting.plotting import Plotter
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


tn = TunnelNetwork(initial_node=False)
z = 0
n_nodes_per_turn = 5
n_turns = 3
r = 50
nodes = []
for i in range(n_turns):
    for j in range(n_nodes_per_turn):
        angle = np.pi * 2 / n_nodes_per_turn * j
        x = np.cos(angle) * r
        y = np.sin(angle) * r
        nodes.append(Node(x, y, z + (np.random.rand(1) - 0.5) * 0))
        z += 3
tunnel = Tunnel(nodes)
tn.add_tunnel(tunnel)
spline = tunnel.spline
n_points = 2000
increment = spline._distance / n_points
ls = [increment * i for i in range(n_points + 1)]
xs = []
ys = []
zs = []
for l in ls:
    p, v = spline(l)
    xs.append(p[0].item())
    ys.append(p[1].item())
    zs.append(p[2].item())
nls = spline._dist_array
nxs = []
nys = []
nzs = []
for node in tunnel.nodes:
    nxs.append(node.x)
    nys.append(node.y)
    nzs.append(node.z)

# PLOT GRAPH
camera = [
    (106.97945165368282, 110.26985745497507, 120.31076060389624),
    (3.0536883140325495, 6.344094115324879, 16.384997264245982),
    (0.0, 0.0, 1.0),
]
plotter = Plotter(off_screen=True)
plotter.set_background("w")
plot_graph(plotter, tn, node_rad=2, edge_rad=0.3)
plotter.add_mesh(pv.Sphere(radius=3, center=nodes[0].xyz), color="r")
plotter.add_mesh(pv.Sphere(radius=3, center=nodes[-1].xyz), color="b")
plotter.camera_position = camera
plotter.show(screenshot=f"/home/lorenzo/images/papers/subt_proc_gen/spline_1.png")
# PLOT SPLINE (NO EDGES)
plotter = Plotter(off_screen=True)
plotter.set_background("w")
plot_spline(plotter, tunnel.spline, color="purple", radius=0.5)
plot_nodes(plotter, tunnel.nodes, radius=2, color="k")
plotter.add_mesh(pv.Sphere(radius=3, center=nodes[0].xyz), color="r")
plotter.add_mesh(pv.Sphere(radius=3, center=nodes[-1].xyz), color="b")
plotter.camera_position = camera
plotter.show(screenshot=f"/home/lorenzo/images/papers/subt_proc_gen/spline_2.png")
# PLOT XYZ components of spline
figsize = (15, 9)
labelpad = 0
fig = plt.figure(figsize=figsize)
ifpsize = 150
psize = 100
fs = 30
# -----
plt.subplot(3, 1, 1)
plt.yticks(fontsize=fs)
plt.scatter(nls, nxs, c="k", zorder=4, s=psize)
plt.scatter(nls[0], nxs[0], c="r", zorder=5, s=ifpsize)
plt.scatter(nls[-1], nxs[-1], c="b", zorder=5, s=ifpsize)
plt.plot(ls, xs, zorder=1)
plt.ylabel("X", fontsize=fs, labelpad=labelpad)
# -----
plt.subplot(3, 1, 2)
plt.yticks(fontsize=fs)
plt.scatter(nls, nys, c="k", zorder=4, s=psize)
plt.scatter(nls[0], nys[0], c="r", zorder=5, s=ifpsize)
plt.scatter(nls[-1], nys[-1], c="b", zorder=5, s=ifpsize)
plt.plot(ls, ys, zorder=1)
plt.ylabel("Y", fontsize=fs, labelpad=labelpad)
# -----
plt.subplot(3, 1, 3)
plt.scatter(nls, nzs, c="k", zorder=4, s=psize)
plt.scatter(nls[0], nzs[0], c="r", zorder=5, s=ifpsize)
plt.scatter(nls[-1], nzs[-1], c="b", zorder=5, s=ifpsize)
plt.plot(ls, zs, zorder=1)
plt.ylabel("Z", fontsize=fs, labelpad=labelpad + 17)
plt.xlabel("Length along the tunnel (l)", fontsize=fs)
fig.subplots_adjust(hspace=0)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig(f"/home/lorenzo/images/papers/subt_proc_gen/spline_3.pdf", dpi=500)
