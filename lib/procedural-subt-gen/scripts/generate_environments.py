import argparse
import os
import pyvista as pv
from pyvista.plotting.plotting import Plotter
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    IntersectionPtClType,
)
from subt_proc_gen.display_functions import *
import numpy as np
import distinctipy
import logging as log

colors = distinctipy.get_colors(30)

log.basicConfig(level=log.DEBUG)

MODEL_SDF_TEXT = """<?xml version="1.0"?>
<sdf version="1.6">
    <model name="tunnel_network">
        <static>true</static>
        <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def gen_axis_points_file(mesh_generator: TunnelNetworkMeshGenerator):
    axis_points = np.zeros((0, 3 + 3 + 1 + 1 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
        aps = mesh_generator.aps_of_tunnels
        avs = mesh_generator.avs_of_tunnels
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        tunnel_flags = np.ones((len(aps), 1)) * 1
        tunnel_id = np.ones((len(aps), 1)) * hash(tunnel)
        axis_points = np.concatenate(
            (axis_points, np.concatenate((aps, avs, rds, tunnel_flags, tunnel_id), axis=1)), axis=0
        )
    for intersection in mesh_generator._tunnel_network.intersections:
        for tunnel in mesh_generator._tunnel_network._tunnels_of_node[intersection]:
            radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
            aps = mesh_generator._aps_avs_of_intersections[intersection][tunnel][:, 0:3]
            avs = mesh_generator._aps_avs_of_intersections[intersection][tunnel][:, 3:6]
            assert len(aps) == len(avs)
            if len(aps) == 0:
                continue
            rds = np.ones((len(aps), 1)) * radius
            intersection_flag = np.ones((len(aps), 1)) * 2
            tunnel_id = np.ones((len(aps), 1)) * hash(tunnel)
            axis_points = np.concatenate(
                (
                    axis_points,
                    np.concatenate((aps, avs, rds, intersection_flag, tunnel_id), axis=1),
                ),
                axis=0,
            )
    return axis_points


def args():
    parser = argparse.ArgumentParser(
        prog="EnvironemtnsGenerator",
        description="This script generates as many environemts as desired, with different topological structures, and saves them in a folder.",
    )
    parser.add_argument("-F", "--folder", type=str, required=True)
    parser.add_argument(
        "-N",
        "--number_of_environments",
        type=int,
        required=True,
        help="Number of environemts that will be generated",
    )
    parser.add_argument(
        "-NGT",
        "--number_of_grown_tunnels",
        default=5,
        required=False,
        type=int,
        help="Number of tunnels that are generated from just an initial node, growing randomly from it.",
    )
    parser.add_argument(
        "-NCT",
        "--number_of_connector_tunnels",
        default=2,
        required=False,
        type=int,
        help="Number of tunnels that are generated from an initial node to a final node already present in the Tunnel Network",
    )
    parser.add_argument(
        "-FTA",
        "--fta_range",
        default=[-2.0, -1.0],
        required=False,
        type=float,
        nargs="*",
        help="This parameter controls what is the minimum vertical distance from the axis of a tunnel to the floor of the tunnel. If this number is positive, the floor of the tunnel will allways be avobe the axis. This number has to be lower thant -FTA",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        default=False,
        action="store_true",
        help="If this is set to True, the environmets previously on the folder will be overwriten",
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="If this is set to True, the environmets previously on the folder will be overwriten",
    )
    return parser.parse_args()


def main():
    arguments = args()
    base_folder = arguments.folder
    n_envs = arguments.number_of_environments
    n_grown = arguments.number_of_grown_tunnels
    n_connector = arguments.number_of_connector_tunnels
    overwrite = arguments.overwrite
    min_fta_distance, max_fta_distance = arguments.fta_range
    plot_after_generation = arguments.plot
    if os.path.isdir(base_folder):
        if not overwrite:
            raise Exception(
                "Folder already exists, if you want to overwrite it set the argument '-O' to 'true'"
            )
        else:
            pass
    os.makedirs(base_folder, exist_ok=True)
    for n in range(n_envs):
        fta_dist = np.random.uniform(min_fta_distance, max_fta_distance)
        base_env_folder = os.path.join(base_folder, f"env_{n+1:03d}")
        os.makedirs(base_env_folder, exist_ok=True)
        tunnel_network_params = TunnelNetworkParams.from_defaults()
        tunnel_network_params.min_distance_between_intersections = 30
        tunnel_network_params.collision_distance = 10
        tunnel_network = TunnelNetwork(params=tunnel_network_params)
        for _ in range(n_grown):
            GrownTunnelGenerationParams._random_distance_range = (100, 300)
            GrownTunnelGenerationParams._random_horizontal_tendency_range_deg = (
                -40,
                40,
            )
            GrownTunnelGenerationParams._random_horizontal_noise_range_deg = (-30, 30)
            GrownTunnelGenerationParams._random_min_segment_length_fraction_range = (
                0.05,
                0.05,
            )
            GrownTunnelGenerationParams._random_max_segment_length_fraction_range = (
                0.10,
                0.10,
            )
            result = False
            while not result:
                params = GrownTunnelGenerationParams.random()
                result = tunnel_network.add_random_grown_tunnel(params=params, n_trials=100)
        for _ in range(n_connector):
            tunnel_network.add_random_connector_tunnel(n_trials=100)
        ptcl_gen_params = TunnelNetworkPtClGenParams.random()
        mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
        mesh_gen_params.fta_distance = fta_dist
        mesh_generator = TunnelNetworkMeshGenerator(
            tunnel_network,
            ptcl_gen_params=ptcl_gen_params,
            meshing_params=mesh_gen_params,
        )
        mesh_generator.compute_all()
        axis_points = gen_axis_points_file(mesh_generator)
        path_to_mesh = os.path.join(base_env_folder, "mesh.obj")
        mesh_generator.save_mesh(path_to_mesh)
        if plot_after_generation:
            plotter = Plotter()
            plot_graph(plotter, tunnel_network)
            mesh = pv.read(path_to_mesh)
            plotter.add_mesh(mesh, style="wireframe")
            plotter.show()
        np.savetxt(os.path.join(base_env_folder, "axis.txt"), axis_points)
        np.savetxt(os.path.join(base_env_folder, "fta_dist.txt"), np.array((fta_dist,)))
        sdf = MODEL_SDF_TEXT.format(path_to_mesh, path_to_mesh)
        path_to_model_sdf = os.path.join(base_env_folder, "model.sdf")

        with open(path_to_model_sdf, "w") as f:
            f.write(sdf)


if __name__ == "__main__":
    main()
