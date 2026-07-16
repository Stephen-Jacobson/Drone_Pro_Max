import argparse
import copy
import os
import shutil
import pyvista as pv
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
    TunnelPtClGenParams,
)
from subt_proc_gen.perlin import CylindricalPerlinNoiseMapperParms
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
            <pose>{}</pose>
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
    axis_points = np.zeros((0, 3 + 3 + 1 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
        aps = mesh_generator.aps_of_tunnels
        avs = mesh_generator.avs_of_tunnels
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        tunnel_flags = np.ones((len(aps), 1)) * 1
        axis_points = np.concatenate(
            (axis_points, np.concatenate((aps, avs, rds, tunnel_flags), axis=1)), axis=0
        )
    for intersection in mesh_generator._tunnel_network.intersections:
        params = mesh_generator.params_of_intersection(intersection)
        if params.ptcl_type == IntersectionPtClType.spherical_cavity:
            radius = params.radius
        elif params.ptcl_type == IntersectionPtClType.no_cavity:
            radiuses = []
            for tunnel in mesh_generator._tunnel_network._tunnels_of_node[intersection]:
                radiuses.append(mesh_generator.ptcl_params_of_tunnel(tunnel).radius)
            radius = max(radiuses)
        else:
            raise NotImplementedError()
        aps = mesh_generator.aps_of_intersection(intersection)
        avs = mesh_generator.avs_of_intersection(intersection)
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        intersection_flags = np.ones((len(aps), 1)) * 2
        axis_points_of_inter = np.concatenate(
            (aps, avs, rds, intersection_flags), axis=1
        )
        axis_points = np.concatenate((axis_points, axis_points_of_inter), axis=0)
    return axis_points


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments_folder_path", type=str, required=True)
    parser.add_argument("--tunnel_lengths", type=float, required=True, nargs="*")
    parser.add_argument("--min_segment_lengths", type=float, required=True, nargs="*")
    parser.add_argument("--max_segment_lengths", type=float, required=True, nargs="*")
    parser.add_argument(
        "--horizontal_tendencies_deg", type=float, required=True, nargs="*"
    )
    parser.add_argument("--horizontal_noises_deg", type=float, required=True, nargs="*")
    parser.add_argument(
        "--vertical_tendencies_deg", type=float, required=True, nargs="*"
    )
    parser.add_argument("--vertical_noises_deg", type=float, required=True, nargs="*")
    parser.add_argument("--radiuses", type=float, required=True, nargs="*")
    parser.add_argument("--roughnesses", type=float, required=True, nargs="*")
    parser.add_argument("--fta_dists", type=float, required=True, nargs="*")
    return parser.parse_args()


def main():
    # Handle arguments
    args = vars(get_args())
    environments_folder_path = args["environments_folder_path"]
    del args["environments_folder_path"]
    os.makedirs(environments_folder_path, exist_ok=True)
    arg_iterator = [{}]
    for arg_name in args:
        arg_vals = args[arg_name]
        new_arg_iterator = []
        for arg_val in arg_vals:
            for prev_arg_it in arg_iterator:
                copy_of_prev_arg_it = copy.deepcopy(prev_arg_it)
                copy_of_prev_arg_it[arg_name] = arg_val
                new_arg_iterator.append(copy_of_prev_arg_it)
        arg_iterator = new_arg_iterator
    print(f"Generating {len(arg_iterator)} envs")
    # Start generating tunnels
    for env_n, args in enumerate(arg_iterator):
        # Extract the arguements
        name = f"env_{env_n:03d}"
        tunnel_length = args["tunnel_lengths"]
        min_segment_length = args["min_segment_lengths"]
        max_segment_length = args["max_segment_lengths"]
        horizontal_tendency_rad = np.deg2rad(args["horizontal_tendencies_deg"])
        horizontal_noise_rad = np.deg2rad(args["horizontal_noises_deg"])
        vertical_tendency_rad = np.deg2rad(args["vertical_tendencies_deg"])
        vertical_noise_rad = np.deg2rad(args["vertical_noises_deg"])
        radius = args["radiuses"]
        roughness = args["roughnesses"]
        fta_dist = args["fta_dists"]
        # Create the directory for the environments
        path_to_env = os.path.join(environments_folder_path, name)
        os.makedirs(path_to_env, exist_ok=True)
        # Create the tunnel network
        tunnel_network_params = TunnelNetworkParams.from_defaults()
        tunnel_network = TunnelNetwork(params=tunnel_network_params)
        grown_tunnel_params = GrownTunnelGenerationParams(
            tunnel_length,
            horizontal_tendency_rad,
            vertical_tendency_rad,
            horizontal_noise_rad,
            vertical_noise_rad,
            min_segment_length,
            max_segment_length,
        )
        result = tunnel_network.add_random_grown_tunnel(
            params=grown_tunnel_params, n_trials=100, yaw_range=(0, 0)
        )
        # Set the parameters for the ptcl generation
        tunnel = list(tunnel_network.tunnels)[-1]
        ptcl_gen_params = TunnelNetworkPtClGenParams.from_defaults()
        tunnel_ptcl_gen_params = TunnelPtClGenParams.from_defaults()
        perlin_params = CylindricalPerlinNoiseMapperParms.from_defaults()
        perlin_params.roughness = roughness
        tunnel_ptcl_gen_params.perlin_params = perlin_params
        tunnel_ptcl_gen_params.radius = radius
        ptcl_gen_params.pre_set_tunnel_params[tunnel] = tunnel_ptcl_gen_params
        mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
        mesh_gen_params.fta_distance = fta_dist
        mesh_generator = TunnelNetworkMeshGenerator(
            tunnel_network,
            ptcl_gen_params=ptcl_gen_params,
            meshing_params=mesh_gen_params,
        )
        mesh_generator.compute_all()
        axis_points = gen_axis_points_file(mesh_generator)
        path_to_mesh = os.path.join(path_to_env, "mesh.obj")
        mesh_generator.save_mesh(path_to_mesh)
        np.savetxt(os.path.join(path_to_env, "axis.txt"), axis_points)
        np.savetxt(os.path.join(path_to_env, "fta_dist.txt"), np.array((fta_dist,)))
        sdf = MODEL_SDF_TEXT.format(path_to_mesh, path_to_mesh)
        path_to_model_sdf = os.path.join(path_to_env, "model.sdf")
        with open(path_to_model_sdf, "w") as f:
            f.write(sdf)


if __name__ == "__main__":
    main()
