import subt_proc_gen.display_functions as display
from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.param_classes import (
    TunnelNetworkPtClGenParams,
)
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
import pyvista as pv
import os
import numpy as np

# PARAMETERS
meshes_save_folder = "/home/lorenzo/Documents/my_papers/ICRA2024_procedural/video/variations_on_same_environment/meshes"
n_meshes = 20
# CODE
os.makedirs(meshes_save_folder, exist_ok=True)
while True:
    tn = TunnelNetwork()
    tn.add_random_grown_tunnel()
    tn.add_random_grown_tunnel()
    tn.add_random_grown_tunnel()
    tn.add_random_grown_tunnel()
    tn.add_random_connector_tunnel(n_trials=1000)
    tn.add_random_connector_tunnel(n_trials=1000)
    mesh_generator = TunnelNetworkMeshGenerator(
        tn, ptcl_gen_params=TunnelNetworkPtClGenParams.random()
    )
    mesh_generator.compute_all()
    plotter = pv.Plotter()
    display.plot_graph(plotter, tn)
    display.plot_splines(plotter, tn)
    display.plot_mesh(plotter, mesh_generator)
    display.plot_ptcl(plotter, mesh_generator.ps)
    plotter.show()
    text = input("Use this network? (y/n): ")
    if "y" in text.lower():
        break

for i in range(n_meshes):
    mesh_generator = TunnelNetworkMeshGenerator(
        tn, ptcl_gen_params=TunnelNetworkPtClGenParams.random()
    )
    mesh_generator.compute_ptcl()
    mesh_generator.compute_mesh()
    if np.random.choice((True, False)):
        mesh_generator.compute_floors()
    mesh_file_name = f"mesh_{i:03d}.obj"
    path_to_mesh = os.path.join(meshes_save_folder, mesh_file_name)
    mesh_generator.save_mesh(path_to_mesh)
