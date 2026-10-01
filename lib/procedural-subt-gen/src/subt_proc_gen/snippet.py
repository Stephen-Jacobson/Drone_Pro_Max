from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from subt_proc_gen.display_functions import plot_mesh, plot_spline, plot_graph
import pyvista as pv
import logging as log
import pickle

log.basicConfig(level=log.DEBUG)
for n in range(10):
    tn = TunnelNetwork()
    n1 = Node(0, 0, 0)
    n2 = Node(25, 0, 0)
    n3 = Node(50, 0, 0)
    n4 = Node(70, 20, 0)
    n5 = Node(50, 40, 0)
    n6 = Node(25, 40, 0)
    n7 = Node(0, 40, 0)
    t1 = Tunnel(nodes=(n1, n2, n3, n4, n5, n6, n7))
    t2 = Tunnel.connector(inode=n2, fnode=n6)
    t3 = Tunnel.connector(inode=n3, fnode=n5)
    t4 = Tunnel.grown(i_node=n6)
    tn.add_tunnel(t1)
    tn.add_tunnel(t2)
    tn.add_tunnel(t3)
    tn.add_tunnel(t4)
    tnmg = TunnelNetworkMeshGenerator(tn)
    tnmg.compute_all()
    tnmg.save_mesh("mesh.obj")
    tnmg.mesh = tnmg.mesh.simplify_vertex_clustering(voxel_size=0.5)
    if n == 0:
        plotter = pv.Plotter()
        plotter.background_color = "w"
        plot_mesh(plotter, tnmg, color="w", style="wireframe")
        plot_spline(plotter, t1.spline, color="g", radius=0.4)
        plot_spline(plotter, t2.spline, color="r", radius=0.4)
        plot_spline(plotter, t3.spline, color="r", radius=0.4)
        plot_spline(plotter, t4.spline, color="b", radius=0.4)
        plotter.show()
    plotter_2 = pv.Plotter(off_screen=True)
    plotter_2.background_color = "w"
    plot_mesh(plotter_2, tnmg, color="w", style="wireframe")
    plot_spline(plotter_2, t1.spline, color="g", radius=0.4)
    plot_spline(plotter_2, t2.spline, color="r", radius=0.4)
    plot_spline(plotter_2, t3.spline, color="r", radius=0.4)
    plot_spline(plotter_2, t4.spline, color="b", radius=0.4)
    if n == 0:
        camera = plotter.camera
    plotter_2.camera = camera
    plotter_2.show(
        screenshot=f"/home/lorenzo/Documents/my_papers/ICRA2024_procedural/paper/figures/snipet_sample_{n}.png",
    )
