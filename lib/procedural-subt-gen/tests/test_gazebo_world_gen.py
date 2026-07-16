from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from subt_proc_gen.gazebo_file_gen import *
import logging
import pyvista as pv
import subt_proc_gen.display_functions as dp

logging.basicConfig(level=logging.DEBUG, force=True)


def test1():
    tunnel_network = TunnelNetwork()
    for i in range(6):
        tunnel_network.add_random_grown_tunnel()
    for i in range(2):
        tunnel_network.add_random_connector_tunnel()
    mesh_generator = TunnelNetworkMeshGenerator(tunnel_network)
    mesh_generator.compute_all()
    mesh_generator_to_gazebo_model(mesh_generator.mesh, "/home/lorenzo/model")


def main():
    for i in range(3):
        test1()


if __name__ == "__main__":
    main()
