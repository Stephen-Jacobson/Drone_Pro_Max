# This file defines functions to create a gazebo world file from a set of meshes
import subt_proc_gen.mesh_generation as mg
import subt_proc_gen.param_classes as mgp
import subt_proc_gen.tunnel as tn
from subt_proc_gen.gazebo_base_sdf_text import *
import open3d as o3d
import os


def format_model_sdf_text(name, x, y, z, r, p, yw, uri, material_text):
    return MODEL_BASE_SDF.format(name, x, y, z, r, p, yw, uri, uri, material_text)


def mesh_generator_to_gazebo_model(
    mesh_generator: mg.TunnelNetworkMeshGenerator, path_to_model_folder
):
    mesh = mesh_generator.mesh
    os.makedirs(path_to_model_folder, exist_ok=True)
    path_to_mesh = os.path.join(path_to_model_folder, "mesh.obj")
    print(mesh.is_edge_manifold())
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tensor_mesh.compute_uvatlas()
    o3d.t.io.write_triangle_mesh(path_to_mesh, tensor_mesh)
    sdf_text = format_model_sdf_text(
        "cave", 0, 0, 0, 0, 0, 0, path_to_mesh, MATERIAL_TEXT
    )
    path_to_sdf = os.path.join(path_to_model_folder, "model.sdf")
    with open(path_to_sdf, "w") as f:
        f.write(sdf_text)
    return path_to_sdf


def mesh_generator_to_gazebo_world(
    mesh_generator: mg.TunnelNetworkMeshGenerator, path_to_folder, number_of_rocks=0
):
    os.makedirs(path_to_folder, exist_ok=True)
    path_to_cave_model_folder = os.path.join(path_to_folder, "cave_model")
    path_to_cave_model_sdf = mesh_generator_to_gazebo_model(
        mesh_generator,
    )
