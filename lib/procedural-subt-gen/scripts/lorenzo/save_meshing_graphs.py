from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh, TunnelPtClGenParams
import pickle
import pathlib
import os
import pickle

# PARAMS
folder_g = "/home/lorenzo/git/procedural-subt-gen/datafiles"
pth_g = os.path.join(folder_g, "graph_for_plotting.pkl")
pth_gnn = os.path.join(folder_g, "graph_for_plotting_no_noise.pkl")
pth_gn = os.path.join(folder_g, "graph_for_plotting_noise.pkl")


def main():
    # Generate the grap
    with open(pth_g, "rb") as f:
        tunnel_network = pickle.load(f)
    params = {}
    params["roughness"] = 0
    params["flatten_floor"] = False
    params["floor_to_axis_distance"] = 1
    params["radius"] = 3
    params = TunnelPtClGenParams(params=params)
    with_mesh_1 = TunnelNetworkWithMesh(
        tunnel_network, i_meshing_params=params)
    params["roughness"] = 0.15
    with_mesh_2 = TunnelNetworkWithMesh(
        tunnel_network, i_meshing_params=params)
    with open(pth_gnn, "wb+") as f:
        pickle.dump(with_mesh_1, f)
    with open(pth_gn, "wb+") as f:
        pickle.dump(with_mesh_2, f)


if __name__ == "__main__":
    main()
