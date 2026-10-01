from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh, TunnelPtClGenParams
import pickle
import pathlib
import os
import pickle
import open3d as o3d
import numpy as np

# PARAMS
base_mesh_folder = (
    "/home/lorenzo/git/procedural-subt-gen/datafiles/variations_on_same_environment"
)


def main():
    # Generate the grap
    tunnel_network = TunnelNetwork()
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    for _ in range(n_grown_tunnels):
        tunnel_network.add_random_tunnel_from_initial_node(trial_limit=200)
    for _ in range(n_connecting_tunnels):
        tunnel_network.add_random_tunnel_from_initial_to_final_node(
            trial_limit=200)
    for n_mesh in range(72):
        tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
        tnm.clean_intersections()
        _, mesh = tnm.compute_mesh()
        filename = f"{n_mesh:04d}.obj"
        mesh_path = os.path.join(base_mesh_folder, filename)
        o3d.io.write_triangle_mesh(mesh_path, mesh)


if __name__ == "__main__":
    main()
