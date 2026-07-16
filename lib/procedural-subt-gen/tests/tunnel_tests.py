from subt_proc_gen.tunnel import (
    TunnelNetwork,
    Tunnel,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.graph import Node
from subt_proc_gen.geometry import Vector3D
from subt_proc_gen.display_functions import (
    plot_nodes,
    plot_edges,
    plot_xyz_axis,
    plot_graph,
    plot_splines,
)
import numpy as np
import os
import pathlib
from time import perf_counter_ns
from traceback import print_exc
import pyvista as pv
import logging as log

log.basicConfig(level=log.DEBUG)


def timeit(function):
    start = perf_counter_ns()
    function()
    end = perf_counter_ns()
    elapsed = (end - start) * 1e-9
    print(f"{function.__name__} took {elapsed:.5f} secs")


def test1():
    values = [0, 1, -1]
    tunnel_network = TunnelNetwork()
    for x in values:
        for y in values:
            for z in values:
                tunnel_network.add_node(Node((x, y, z)))


def test2():
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    tunnel_network.add_tunnel(
        Tunnel.grown(
            i_node=first_node,
            i_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(0), length=30
            ),
            params=GrownTunnelGenerationParams.from_defaults(),
        )
    )
    params = GrownTunnelGenerationParams.random()
    tunnel_network.add_tunnel(
        Tunnel.grown(
            i_node=first_node,
            i_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(0), length=30
            ),
            params=params,
        )
    )


def test3():
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(0), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(90), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        inode=first_tunnel[-2],
        fnode=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.remove_tunnel(first_tunnel)
    tunnel_network.add_tunnel(third_tunnel)


def test4():
    """Tests the serializtion and deserialization from yaml"""
    tunnel_network = TunnelNetwork(initial_node=False)
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(0), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(90), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        inode=first_tunnel[-2],
        fnode=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(third_tunnel)
    path_to_save_file = os.path.join(pathlib.Path.home(), "test.yaml")
    tunnel_network.to_yaml(path_to_save_file)
    del tunnel_network
    tunnel_network = TunnelNetwork()
    tunnel_network.load_yaml(path_to_save_file)
    os.remove(path_to_save_file)


def test5():
    """Tests the node classification"""
    tunnel_network = TunnelNetwork(initial_node=False)
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(0), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(90), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        inode=first_tunnel[-2],
        fnode=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(third_tunnel)
    tunnel_network.compute_node_types()


def test6():
    """Tests the intersection connections"""
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(0), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(90), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        inode=first_tunnel[-2],
        fnode=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(third_tunnel)
    intersection_connections = tunnel_network.compute_intersection_connectivity_graph()


def test7():
    tunnel_network = TunnelNetwork()
    for i in range(10):
        print(i, end="\r", flush=True)
        result = False
        while not result:
            result = tunnel_network.add_random_grown_tunnel()
    for i in range(10):
        print(i, end="\r", flush=True)
        result = False
        while not result:
            result = tunnel_network.add_random_connector_tunnel()
    plotter = pv.Plotter()
    plot_graph(plotter, tunnel_network, edge_color="g")
    plot_splines(plotter, tunnel_network)
    plot_xyz_axis(plotter)
    plotter.show()


def main():
    tests = [test1, test2, test3, test4, test5, test6, test7]
    for test in tests:
        try:
            timeit(test)
        except:
            print(f"{test.__name__} failed")
            print_exc()


if __name__ == "__main__":
    main()
