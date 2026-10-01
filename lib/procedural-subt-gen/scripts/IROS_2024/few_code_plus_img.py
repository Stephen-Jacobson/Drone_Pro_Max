from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from subt_proc_gen.display_functions import plot_xyz_axis
import pyvista as pv

tn = TunnelNetwork()
while True:
    tn.add_random_grown_tunnel(n_trials=1)
    if len(tn.tunnels) == 5:
        break
while True:
    tn.add_random_connector_tunnel(n_trials=1)
    if len(tn.tunnels) >= 10:
        break
tnmg = TunnelNetworkMeshGenerator(tn)
tnmg.compute_all()
