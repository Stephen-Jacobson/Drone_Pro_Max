from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, TunnelNetworkParams, GrownTunnelGenerationParams, ConnectorTunnelGenerationParams
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator, TunnelNetworkPtClGenParams, TunnelNetworkMeshGenParams
from subt_proc_gen.display_functions import plot_graph, plot_splines, plot_ptcl, plot_mesh, plot_xyz_axis, plot_nodes
import numpy as np
import pyvista as pv
from pyvista import Plotter
import os

IMAGE_SAVE_FOLDER = "/home/lorenzo/Documents/conferences/IROS_2024"
pv.rcParams["transparent_background"] = True


def gen_tn():
    # Generate the tunnel network
    tn_params = TunnelNetworkParams.from_defaults()
    tn = TunnelNetwork(tn_params)
    gtpm = GrownTunnelGenerationParams.from_defaults()
    gtpm.distance = 100
    gtpm.horizontal_tendency_rad = np.deg2rad(10)
    gtpm.horizontal_noise_rad = np.deg2rad(20)
    tn.add_random_grown_tunnel(gtpm)
    gtpm = GrownTunnelGenerationParams.from_defaults()
    gtpm.distance = 50
    gtpm.horizontal_tendency_rad = np.deg2rad(-10)
    gtpm.horizontal_noise_rad = np.deg2rad(20)
    tn.add_random_grown_tunnel(gtpm)
    ctpm = ConnectorTunnelGenerationParams.from_defaults()
    ctpm.node_position_horizontal_noise = 4
    ctpm.node_position_vertical_noise = 3
    ctpm.segment_length = 30
    tn.add_random_connector_tunnel(n_trials=20)
    tn.add_random_connector_tunnel(n_trials=20)
    tn.add_random_connector_tunnel(n_trials=20)
    tn.add_random_connector_tunnel(n_trials=20)
    return tn


def gen_tnmg(tn):
    tnmg = TunnelNetworkMeshGenerator(tn)
    return tnmg


def _graph_plot_set_camera(tn):
    plotter = Plotter(off_screen=False)
    plot_graph(plotter, tn, edge_color="gray")
    plotter.show()
    return plotter.camera_position


def _graph_plot_capture(tn, camera):
    plotter = Plotter(off_screen=True)
    plot_graph(plotter, tn)
    file = os.path.join(IMAGE_SAVE_FOLDER, "s2_a_graph.png")
    plotter.camera_position = camera
    plotter.show(screenshot=file)


def _splines_plot_capture(tn: TunnelNetwork, camera):
    plotter = Plotter(off_screen=True)
    plot_nodes(plotter, tn.nodes)
    plot_splines(plotter, tn, radius=0.5)
    plotter.camera_position = camera
    file = os.path.join(IMAGE_SAVE_FOLDER, "s2_b_splines.png")
    plotter.show(screenshot=file)


def _ptcl_plot(tnmg: TunnelNetworkMeshGenerator, camera):
    plotter = Plotter(off_screen=True)
    plot_splines(plotter, tnmg._tunnel_network, radius=0.5)
    plot_ptcl(plotter, tnmg.ptcl, radius=0.1, color="red")
    plotter.camera_position = camera
    file = os.path.join(IMAGE_SAVE_FOLDER, "s2_c_ptcl.png")
    plotter.show(screenshot=file)


def _mesh_plot(tnmg: TunnelNetworkMeshGenerator, camera, fname):
    plotter = Plotter(off_screen=True)
    plot_splines(plotter, tnmg._tunnel_network, radius=0.5)
    plot_mesh(plotter, tnmg, color="white", style="wireframe")
    plotter.camera_position = camera
    file = os.path.join(IMAGE_SAVE_FOLDER, fname)
    plotter.show(screenshot=file)


def main():
    tn = gen_tn()
    print("tn generated")
    camera = _graph_plot_set_camera(tn)
    _graph_plot_capture(tn, camera)
    _splines_plot_capture(tn, camera)
    tnmg = gen_tnmg(tn)
    tnmg.compute_ptcl()
    _ptcl_plot(tnmg, camera)
    tnmg.compute_mesh()
    _mesh_plot(tnmg, camera, "s2_d_mesh.png")
    tnmg.add_noise_to_mesh()
    _mesh_plot(tnmg, camera, "s2_e_mesh.png")
    tnmg.compute_floors()
    _mesh_plot(tnmg, camera, "s2_f_mesh.png")


if __name__ == "__main__":
    main()
