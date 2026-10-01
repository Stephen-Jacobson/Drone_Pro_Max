from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator, TunnelPtClGenParams, TunnelNetworkMeshGenParams, TunnelNetworkPtClGenParams
import pyvista as pv

base_folder = "/home/lorenzo/Documents/conferences/IROS_2024/presentacion/data/different_tunnels"

radiuses = [2, 5, 10]
floors = [-1, -2]
roughnesses = [0.00001, 1]

tn = TunnelNetwork()
tn.add_random_grown_tunnel()
tunnel = list(tn.tunnels)[0]

for r in radiuses:
    for f in floors:
        for rough in roughnesses:
            tptclpar = TunnelPtClGenParams.from_defaults()
            tptclpar.noise_multiplier = rough
            tptclpar.radius = r
            tnptclpar = TunnelNetworkPtClGenParams.from_defaults()
            tnptclpar.pre_set_tunnel_params = {tunnel: tptclpar}
            tnmg = TunnelNetworkMeshGenerator(tn, ptcl_gen_params=tnptclpar)
            tnmg._meshing_params.fta_distance = f
            tnmg._meshing_params.poisson_depth = 10
            tnmg.compute_all()
            mesh = tnmg.pyvista_mesh
            filename = f"rad_{r}_floor_{f}_rough_{rough}.stl"
            pv.save_meshio(filename, mesh)
