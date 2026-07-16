from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
)
from subt_proc_gen.perlin import (
    CylindricalPerlinNoiseGenerator,
    CylindricalPerlinNoiseMapperParms,
)
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    Tunnel,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.graph import Node
from subt_proc_gen.geometry import Point3D, Vector3D, get_two_perpendicular_vectors
from subt_proc_gen.display_functions import *
import numpy as np
from multiprocessing import Pool
from time import perf_counter_ns
from traceback import print_exc
import pyvista as pv
import logging as log
import distinctipy
import os

log.basicConfig(level=log.DEBUG)

colors = distinctipy.get_colors(30)


def timeit(function, **args):
    start = perf_counter_ns()
    result = function(**args)
    end = perf_counter_ns()
    elapsed = (end - start) * 1e-9
    print(f"{function.__name__} took {elapsed:.5f} secs")
    return result


def generate_coords(scale):
    coords = [] * scale**2
    for i in range(scale):
        for j in range(scale):
            coords.append(np.array((i, j)))
    return coords


def noise_with_pool(generator: CylindricalPerlinNoiseGenerator, coords):
    return generator(coords)


def fill_image_with_noise(image, noise, coords):
    for n, c in zip(noise, coords):
        image[c[0], c[1]] = n
    return image


def test1():
    scale = 100
    generator = CylindricalPerlinNoiseGenerator(
        scale, CylindricalPerlinNoiseMapperParms.from_defaults()
    )
    coords = generate_coords(scale=scale)
    noise = noise_with_pool(generator=generator, coords=coords)
    image = np.zeros((scale, scale))
    image = fill_image_with_noise(image=image, noise=noise, coords=coords)


def test2():
    tunnel_network = Tunnel
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
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=TunnelNetworkPtClGenParams.random(),
        meshing_params=TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()
    mesh_generator.save_mesh("test_2.obj")
    os.remove("test_2.obj")
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(mesh_generator.ps))
    plotter.show()


def test3():
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
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=TunnelNetworkPtClGenParams.from_defaults(),
        meshing_params=TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()
    mesh_generator.save_mesh("mesh.obj")
    os.remove("mesh.obj")
    print(len(mesh_generator.mesh.points))


def test4():
    tunnel_network = TunnelNetwork()
    tunnel_network.params.min_distance_between_intersections = 40
    tunnel_network.params.collision_distance = 10
    for i in range(3):
        print(i, end="\r", flush=True)
        result = False
        while not result:
            result = tunnel_network.add_random_grown_tunnel()
    for i in range(2):
        print(i, end="\r", flush=True)
        result = False
        while not result:
            result = tunnel_network.add_random_connector_tunnel()
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=TunnelNetworkPtClGenParams.random(),
        meshing_params=TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()
    plotter = pv.Plotter()
    for tunnel in mesh_generator._ptcl_of_tunnels:
        if len(mesh_generator.ps_of_tunnel(tunnel)) > 0:
            plotter.add_mesh(
                pv.PolyData(mesh_generator.ps_of_tunnel(tunnel)), color="b"
            )
    for i, intersection in enumerate(mesh_generator._original_ptcl_of_intersections):
        if len(mesh_generator.ps_of_intersection(intersection)) > 0:
            plotter.add_mesh(
                pv.PolyData(mesh_generator.ps_of_intersection(intersection)),
                color=colors[i],
            )
    plotter.add_mesh(mesh_generator.pyvista_mesh)
    plotter.show()
    mesh_generator.save_mesh("mesh.obj")
    os.remove("mesh.obj")


def test5():
    tunnel_network = TunnelNetwork(initial_node=False)
    nodes = (
        Node((-20, 0, 0)),
        Node((0, 0, 0)),
        Node((20, 0, 0)),
        Node((20, 20, 0)),
        Node((0, 20, 0)),
        Node((0, 0, 0)),
        Node((0, -20, 0)),
    )
    tunnel_network.add_tunnel(Tunnel(nodes))
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        TunnelNetworkPtClGenParams.from_defaults(),
        TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_generator.pyvista_mesh, style="wireframe")
    plotter.add_mesh(pv.PolyData(mesh_generator.ps))
    plot_tunnel_network_graph(plotter, tunnel_network)
    plotter.show()


def main():
    tests = [test1, test2, test3, test4, test5]
    for test in tests:
        try:
            timeit(test)
        except:
            print(f"{test.__name__} failed")
            print_exc()


if __name__ == "__main__":
    main()
