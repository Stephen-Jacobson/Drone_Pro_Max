import numpy as np
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.geometry import (
    get_two_perpendicular_vectors,
    get_close_points_indices,
    get_close_points_to_point,
    get_uniform_points_in_sphere,
    Point3D,
    warp_angle_pi,
    distance_matrix,
)
from subt_proc_gen.perlin import (
    CylindricalPerlinNoiseGenerator,
    SphericalPerlinNoiseMapper,
)
from subt_proc_gen.param_classes import (
    TunnelNetworkPtClGenStrategies,
    TunnelPtClGenParams,
    IntersectionPtClType,
    IntersectionPtClGenParams,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
)
from tqdm import tqdm
from enum import Enum
import logging as log
import pyvista as pv
import open3d as o3d
import os
import math
import trace

# TODO: When assigning parameters to the IntersectionPtclGenration step, ensure that diaphanous intersections wont be created if one of the tunnels in that direction is not long enough


class PtclVoxelizator:
    def __init__(self, ptcl: np.ndarray, voxel_size=5, coords_range=[0, 3]):
        self.ptcl = ptcl
        self.voxel_size = voxel_size
        self.grid = dict()
        ijks = self.xyz_to_ijk(ptcl[:, coords_range[0] : coords_range[1]])
        self.ncols = ptcl.shape[1]
        unique_ijks = set()
        for ijk in ijks:
            i, j, k = ijk
            unique_ijks.add((i, j, k))
        self.unique_ijks = unique_ijks
        for ijk in tqdm(unique_ijks, total=len(unique_ijks), desc="Voxelizing floors"):
            ijk = np.array((ijk,))
            idxs = np.where(np.prod(ijks == ijk, axis=1))
            i, j, k = ijk[0]
            self.grid[(i, j, k)] = np.copy(self.ptcl[idxs, :][0])
            self.ptcl = np.delete(self.ptcl, idxs, axis=0)
            ijks = np.delete(ijks, idxs, axis=0)
        assert len(self.ptcl) == 0

    def xyz_to_ijk(self, coords):
        return np.floor(coords / self.voxel_size).astype(int)

    def get_relevant_points(self, xyz):
        _i, _j, _k = self.xyz_to_ijk(xyz)
        relevant_points = []
        for i in (_i - 1, _i, _i + 1):
            for j in (_j - 1, _j, _j + 1):
                for k in (_k - 1, _k, _k + 1):
                    if (i, j, k) in self.grid:
                        relevant_points.append(self.grid[(i, j, k)])
        if len(relevant_points) == 0:
            return None
        relevant_points = np.vstack(relevant_points)
        return relevant_points


class TunnelNetworkMeshGenerator:
    def __init__(
        self,
        tunnel_network: TunnelNetwork,
        ptcl_gen_params: TunnelNetworkPtClGenParams = None,
        meshing_params: TunnelNetworkMeshGenParams = None,
    ):
        if ptcl_gen_params is None:
            ptcl_gen_params = TunnelNetworkPtClGenParams.random()
        if meshing_params is None:
            meshing_params = TunnelNetworkMeshGenParams.from_defaults()
        self._tunnel_network = tunnel_network
        self._ptcl_gen_params = ptcl_gen_params
        self._meshing_params = meshing_params
        self._params_of_intersections = dict()
        self._params_of_tunnels = dict()
        self._perlin_generator_of_tunnel = dict()
        self._ptcl_of_tunnels = dict()
        self._aps_avs_of_tunnels = dict()
        self._ptcl_of_intersections = dict()
        self._aps_avs_of_intersections = dict()
        self._radius_of_intersections_for_tunnels = dict()
        self._original_ptcl_of_intersections = dict()
        self._original_aps_avs_of_intersections = dict()
        self._voxelized_ptcl = None

    def ps_of_tunnel(self, tunnel: Tunnel):
        return self._ptcl_of_tunnels[tunnel][:, :3]

    def ns_of_tunnel(self, tunnel: Tunnel):
        return self._ptcl_of_tunnels[tunnel][:, 3:6]

    def apss_of_tunnel(self, tunnel: Tunnel):
        return self._ptcl_of_tunnels[tunnel][:, 6:9]

    def avss_of_tunnel(self, tunnel: Tunnel):
        return self._ptcl_of_tunnels[tunnel][:, 9:12]

    def aps_of_tunnel(self, tunnel: Tunnel):
        return self._aps_avs_of_tunnels[tunnel][:, :3]

    def avs_of_tunnel(self, tunnel: Tunnel):
        return self._aps_avs_of_tunnels[tunnel][:, 3:]

    def ptcl_of_intersection(self, intersection: Node):
        return np.vstack([self._ptcl_of_intersections[intersection][element] for element in self._ptcl_of_intersections[intersection]])

    def ps_of_intersection(self, intersection: Node):
        return self.ptcl_of_intersection(intersection)[:, 0:3]

    def ns_of_intersection(self, intersection: Node):
        return self.ptcl_of_intersection(intersection)[:, 3:6]

    def apss_of_intersection(self, intersection: Node):
        return self.ptcl_of_intersection(intersection)[:, 6:9]

    def avss_of_intersection(self, intersection: Node):
        return self.ptcl_of_intersection(intersection)[:, 9:12]

    def aps_of_intersection(self, intersection: Node):
        to_return = np.zeros((0, 3))
        for element in self._aps_avs_of_intersections[intersection]:
            to_return = np.vstack(
                (
                    to_return,
                    self._aps_avs_of_intersections[intersection][element][:, 0:3],
                )
            )
        return to_return

    def avs_of_intersection(self, intersection: Node):
        to_return = np.zeros((0, 3))
        for element in self._aps_avs_of_intersections[intersection]:
            to_return = np.vstack(
                (
                    to_return,
                    self._aps_avs_of_intersections[intersection][element][:, 3:6],
                )
            )
        return to_return

    @property
    def tunnels(self):
        return self._tunnel_network.tunnels

    @property
    def intersections(self):
        return self._tunnel_network.intersections

    @property
    def combined_tunnel_ptcls(self):
        return np.vstack([self._ptcl_of_tunnels[tunnel] for tunnel in self.tunnels])

    @property
    def combined_intersection_ptcls(self):
        return (
            np.vstack([self.ptcl_of_intersection(intersection) for intersection in self.intersections])
            if len(self.intersections) > 0
            else np.zeros((0, 12))
        )

    @property
    def ptcl(self):
        return np.vstack([self.combined_tunnel_ptcls, self.combined_intersection_ptcls])

    @property
    def ps(self):
        return self.ptcl[:, 0:3]

    @property
    def ns(self):
        return self.ptcl[:, 3:6]

    @property
    def apss(self):
        return self.ptcl[:, 6:9]

    @property
    def avss(self):
        return self.ptcl[:, 9:12]

    @property
    def aps_of_tunnels(self):
        return np.vstack([self.aps_of_tunnel(tunnel) for tunnel in self.tunnels])

    @property
    def avs_of_tunnels(self):
        return np.vstack([self.avs_of_tunnel(tunnel) for tunnel in self.tunnels])

    @property
    def aps_of_intersections(self):
        return (
            np.vstack([self.aps_of_intersction(intersection) for intersection in self.intersections])
            if len(self.intersections) > 0
            else np.zeros((0, 3))
        )

    @property
    def avs_of_intersections(self):
        return (
            np.vstack([self.avs_of_intersction(intersection) for intersection in self.intersections])
            if len(self.intersections) > 0
            else np.zeros((0, 3))
        )

    @property
    def aps(self):
        return np.vstack([self.aps_of_tunnels, self.aps_of_intersections])

    @property
    def avs(self):
        return np.vstack([self.avs_of_tunnels, self.avs_of_intersections])

    def params_of_intersection(self, intersection) -> IntersectionPtClGenParams:
        return self._params_of_intersections[intersection]

    def ptcl_params_of_tunnel(self, tunnel) -> TunnelPtClGenParams:
        return self._params_of_tunnels[tunnel]

    def _set_params_of_each_tunnel_ptcl_gen(self, ptcl_gen_params=None):
        if ptcl_gen_params is None:
            ptcl_gen_params = self._ptcl_gen_params
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.default:
            for tunnel in self._tunnel_network.tunnels:
                self._params_of_tunnels[tunnel] = (
                    ptcl_gen_params.pre_set_tunnel_params[tunnel]
                    if tunnel in ptcl_gen_params.pre_set_tunnel_params
                    else TunnelPtClGenParams.from_defaults()
                )
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.random:
            for tunnel in self._tunnel_network.tunnels:
                self._params_of_tunnels[tunnel] = (
                    ptcl_gen_params.pre_set_tunnel_params[tunnel] if tunnel in ptcl_gen_params.pre_set_tunnel_params else TunnelPtClGenParams.random()
                )

    def _set_params_of_each_intersection_ptcl_gen(self, ptcl_gen_params: TunnelNetworkPtClGenParams = None):
        if ptcl_gen_params is None:
            ptcl_gen_params = self._ptcl_gen_params
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.default:
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[intersection] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[intersection] = IntersectionPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.random:
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[intersection] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[intersection] = IntersectionPtClGenParams.random()

    def _set_perlin_mappers_of_tunnels(self):
        for tunnel in self._tunnel_network.tunnels:
            self._perlin_generator_of_tunnel[tunnel] = CylindricalPerlinNoiseGenerator(
                length=tunnel.spline.metric_length,
                radius=self.ptcl_params_of_tunnel(tunnel).radius,
                perlin_params=self.ptcl_params_of_tunnel(tunnel).perlin_params,
            )

    def _compute_tunnel_ptcls(self):
        for tunnel in self._tunnel_network.tunnels:
            (
                ptcl,
                normals,
                aps,
                avs,
                apss,
                avss,
            ) = ptcl_from_tunnel(
                tunnel=tunnel,
                perlin_generator=self._perlin_generator_of_tunnel[tunnel],
                dist_between_circles=self.ptcl_params_of_tunnel(tunnel).dist_between_circles,
                n_points_per_circle=self.ptcl_params_of_tunnel(tunnel).n_points_per_circle,
                radius=self.ptcl_params_of_tunnel(tunnel).radius,
                noise_magnitude=self.ptcl_params_of_tunnel(tunnel).noise_multiplier,
            )
            self._ptcl_of_tunnels[tunnel] = np.hstack((ptcl, normals, apss, avss))
            self._aps_avs_of_tunnels[tunnel] = np.hstack((aps, avs))

    def _compute_radius_of_intersections_for_all_tunnels(self):
        for intersection in self._tunnel_network.intersections:
            self._radius_of_intersections_for_tunnels[intersection] = dict()
            for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                intersection_radius_for_tunnel_at_intersection = 0
                for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                    # Tunnel i is the one being evaluated here
                    if tunnel_i is tunnel_j:
                        continue
                    intersection_radius_for_tunnel_at_intersection = max(
                        intersection_radius_for_tunnel_at_intersection,
                        self.get_safe_radius_from_intersection_of_two_tunnels(
                            tunnel_i,
                            tunnel_j,
                            intersection,
                            starting_radius=self.params_of_intersection(intersection).radius,
                        ),
                    )
                self._radius_of_intersections_for_tunnels[intersection][tunnel_i] = intersection_radius_for_tunnel_at_intersection

    def _separate_intersection_ptcl_from_tunnel_ptcls(self):
        for intersection in self._tunnel_network.intersections:
            tunnels_of_intersection = self._tunnel_network._tunnels_of_node[intersection]
            ptcl_of_intersection = dict()
            aps_avs_of_intersection = dict()
            for tunnel in tunnels_of_intersection:
                taps = self.aps_of_tunnel(tunnel)
                tapss = self.apss_of_tunnel(tunnel)
                radius = self._radius_of_intersections_for_tunnels[intersection][tunnel]
                idxs_of_aps_from_t_to_int = get_close_points_indices(
                    intersection.xyz,
                    taps,
                    radius,
                )
                idxs_of_ptcl_from_t_to_int = get_close_points_indices(intersection.xyz, tapss, radius)
                ptcl_of_intersection[tunnel] = np.reshape(
                    self._ptcl_of_tunnels[tunnel][idxs_of_ptcl_from_t_to_int, :],
                    (-1, 12),
                )
                aps_avs_of_intersection[tunnel] = np.reshape(
                    self._aps_avs_of_tunnels[tunnel][idxs_of_aps_from_t_to_int, :],
                    (-1, 6),
                )
                self._ptcl_of_tunnels[tunnel] = np.delete(self._ptcl_of_tunnels[tunnel], idxs_of_ptcl_from_t_to_int, axis=0)
                self._aps_avs_of_tunnels[tunnel] = np.delete(self._aps_avs_of_tunnels[tunnel], idxs_of_aps_from_t_to_int, axis=0)
            self._ptcl_of_intersections[intersection] = ptcl_of_intersection
            self._aps_avs_of_intersections[intersection] = aps_avs_of_intersection
        # Save the original points for the floor flattening step
        for intersection in self.intersections:
            self._original_aps_avs_of_intersections[intersection] = dict()
            self._original_ptcl_of_intersections[intersection] = dict()
            for element in self._aps_avs_of_intersections[intersection]:
                self._original_aps_avs_of_intersections[intersection][element] = np.copy(self._aps_avs_of_intersections[intersection][element])
            for element in self._ptcl_of_intersections[intersection]:
                self._original_ptcl_of_intersections[intersection][element] = np.copy(self._ptcl_of_intersections[intersection][element])

    def clean_tunnels_in_intersection(self, intersection: Node):
        # Get the ids of the points to delete
        idxs_of_tunnel_pts_to_delete = dict()
        for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
            ids_to_delete_from_i = set()
            for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                if tunnel_i is tunnel_j:
                    continue
                ids_of_i_in_j = np.reshape(
                    points_inside_of_tunnel_section(
                        self._aps_avs_of_intersections[intersection][tunnel_j][:, :3],
                        self._ptcl_of_intersections[intersection][tunnel_j][:, :3],
                        self._ptcl_of_intersections[intersection][tunnel_i][:, :3],
                        self._aps_avs_of_intersections[intersection][tunnel_j][:, 3:6],
                    ),
                    -1,
                )
                for idx in ids_of_i_in_j:
                    ids_to_delete_from_i.add(idx)
            idxs_of_tunnel_pts_to_delete[tunnel_i] = np.array(list(ids_to_delete_from_i))
        for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
            if len(idxs_of_tunnel_pts_to_delete[tunnel]) == 0:
                continue
            self._ptcl_of_intersections[intersection][tunnel] = np.delete(
                self._ptcl_of_intersections[intersection][tunnel],
                idxs_of_tunnel_pts_to_delete[tunnel],
                axis=0,
            )

    def _compute_all_intersections_ptcl(self):
        for intersection in self._tunnel_network.intersections:
            params = self.params_of_intersection(intersection)
            if params.ptcl_type == IntersectionPtClType.no_cavity:
                self.clean_tunnels_in_intersection(intersection)
            elif params.ptcl_type == IntersectionPtClType.spherical_cavity:
                self.clean_tunnels_in_intersection(intersection)
                center_point = intersection.xyz
                radius = params.radius
                points_per_sm = params.points_per_sm
                # Generate the points of the intersection
                sphere_ps, sphere_ns = generate_noisy_sphere(
                    center_point,
                    radius,
                    points_per_sm,
                    SphericalPerlinNoiseMapper(params.perlin_params),
                    params.noise_multiplier,
                )
                ids_to_delete_in_cavity = set()
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                    axis_of_tunnel = self._aps_avs_of_intersections[intersection][tunnel][:, 0:3]
                    points_of_tunnel = self._ptcl_of_intersections[intersection][tunnel][:, 0:3]
                    ids_to_delete_in_tunnel = ids_points_inside_ptcl_sphere(sphere_ps, center_point, points_of_tunnel)
                    idxs_to_delete_in_cavity_ = points_inside_of_tunnel_section(axis_of_tunnel, points_of_tunnel, sphere_ps)
                    self._ptcl_of_intersections[intersection][tunnel] = np.delete(
                        self._ptcl_of_intersections[intersection][tunnel],
                        ids_to_delete_in_tunnel,
                        axis=0,
                    )
                    for idx in np.reshape(idxs_to_delete_in_cavity_, -1):
                        ids_to_delete_in_cavity.add(idx)
                assert len(sphere_ps) == len(sphere_ns)
                sphere_ps = np.delete(sphere_ps, np.array(list(ids_to_delete_in_cavity)), axis=0)
                sphere_ns = np.delete(sphere_ns, np.array(list(ids_to_delete_in_cavity)), axis=0)
                assert len(sphere_ps) == len(sphere_ns)
                sphere_apss = np.ones(sphere_ps.shape) * center_point
                sphere_avss = np.zeros(sphere_ps.shape)
                self._ptcl_of_intersections[intersection]["cavity"] = np.hstack(
                    (
                        sphere_ps,
                        sphere_ns,
                        sphere_apss,
                        sphere_avss,
                    )
                )

    def get_safe_radius_from_intersection_of_two_tunnels(
        self,
        tunnel_i: Tunnel,
        tunnel_j: Tunnel,
        intersection: Node,
        increment=2,
        starting_radius=20,
    ):
        assert intersection in tunnel_i.nodes
        assert intersection in tunnel_j.nodes
        tpi = self.ps_of_tunnel(tunnel_i)
        tpj = self.ps_of_tunnel(tunnel_j)
        apj = self.aps_of_tunnel(tunnel_j)
        cut_off_radius = starting_radius
        while True:
            aidx = get_close_points_indices(intersection.xyz, apj, cut_off_radius)
            apj_to_use = np.reshape(apj[aidx, :], (-1, 3))
            tpj_to_use = get_close_points_to_point(intersection.xyz, tpj, cut_off_radius)
            tpi_to_use = get_close_points_to_point(intersection.xyz, tpi, cut_off_radius)
            id_of_p_of_i_inside_j = points_inside_of_tunnel_section(apj_to_use, tpj_to_use, tpi_to_use)
            p_of_i_inside_j = np.reshape(tpi_to_use[id_of_p_of_i_inside_j, :], (-1, 3))
            dist_of_i_points_inside_j_to_inter = np.linalg.norm(intersection.xyz - p_of_i_inside_j, axis=1)
            if len(dist_of_i_points_inside_j_to_inter) == 0:
                radius = cut_off_radius
                break
            radius = np.max(dist_of_i_points_inside_j_to_inter)
            if radius < cut_off_radius - increment:
                break
            cut_off_radius += increment
        return max(radius + increment, starting_radius)

    def compute_ptcl(self):
        log.info("Setting parameters of tunnels")
        self._set_params_of_each_tunnel_ptcl_gen()
        log.info("Setting perlin mappers for tunnels")
        self._set_perlin_mappers_of_tunnels()
        log.info("Computing pointclouds of tunnels")
        self._compute_tunnel_ptcls()
        log.info("Setting paramters for intersections")
        self._set_params_of_each_intersection_ptcl_gen()
        log.info("Calcludating radius of intersections")
        self._compute_radius_of_intersections_for_all_tunnels()
        log.info("Separating the pointclouds of the tunnels for the intersections")
        self._separate_intersection_ptcl_from_tunnel_ptcls()
        log.info("Computing pointclouds of intersections")
        self._compute_all_intersections_ptcl()

    def compute_mesh(self):
        log.info("Computing mesh")
        self._compute_mesh()

    def compute_floors(self):
        log.info("Voxelizing ptcl")
        self._voxelize_ptcl()
        log.info("Flattening floors")
        self._flatten_floors()

    def compute_all(self, compute_floors=True):
        self.compute_ptcl()
        self.compute_mesh()
        self.add_noise_to_mesh()
        if compute_floors:
            self.compute_floors()

    def add_noise_to_mesh(self):
        vertices = np.asarray(self.mesh.vertices)
        vertices += np.random.uniform(-0.2, 0.2, vertices.shape)
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)

    def _compute_mesh(self):
        points = self.ps
        normals = -self.ns
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self._meshing_params.poisson_depth)
        self.mesh = o3d_mesh.simplify_vertex_clustering(voxel_size=0.5)

    def flip_mesh_normals(self):
        pv_mesh = self.pyvista_mesh
        pv_mesh.flip_normals()
        pv.save_meshio("temp.ply", pv_mesh)
        self.mesh = o3d.io.read_triangle_mesh("temp.ply")
        os.remove("temp.ply")

    def _voxelize_ptcl(self):
        self._voxelized_ptcl = PtclVoxelizator(self.ptcl, voxel_size=self._meshing_params.voxelization_voxel_size)

    def _flatten_floors(self):
        # TODO: No longer use radius in smoothing, use the connected vertices
        vertices = np.asarray(self.mesh.vertices)
        fta_dist = self._meshing_params.fta_distance
        floor_vertices_idxs = set()
        pb = tqdm(total=len(vertices), desc="Checking wich vertices are floors")
        for i in range(len(vertices)):
            vert_inside = vertices[i, :]
            local_ptcl = self._voxelized_ptcl.get_relevant_points(vert_inside)
            if local_ptcl is None:
                continue
            aps = local_ptcl[:, 6:9]
            ps = local_ptcl[:, 0:3]
            ap_of_vert = aps[np.argmin(np.linalg.norm(ps - vert_inside, axis=1)), :]
            if vert_inside[2] - ap_of_vert[2] < fta_dist:
                vert_inside[2] = ap_of_vert[2] + fta_dist
                vertices[i, :] = vert_inside
                floor_vertices_idxs.add(i)
            pb.update(1)
        floor_vertices_idxs = tuple(floor_vertices_idxs)
        # Extra steps for diaphanous intersections
        floor_vertices = vertices[floor_vertices_idxs, :]
        floor_vertices = np.reshape(floor_vertices, (-1, 3))
        pb = tqdm(total=len(self.intersections), desc="Checking wich intersection vertices are floors")
        for n_intersection, intersection in enumerate(self.intersections):
            pb.update(1)
            if not (self.params_of_intersection(intersection).ptcl_type == IntersectionPtClType.spherical_cavity):
                continue
            intersection_floor_vertices_idxs = np.where(
                np.linalg.norm(floor_vertices - intersection.xyz, axis=1) < self.params_of_intersection(intersection).radius
            )
            int_floor_verts = np.reshape(floor_vertices[intersection_floor_vertices_idxs, :], (-1, 3))
            for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                aps = self._original_aps_avs_of_intersections[intersection][tunnel][:, :3]
                tps = self._original_ptcl_of_intersections[intersection][tunnel][:, :3]
                aps = np.reshape(aps, (-1, 3))
                tps = np.reshape(tps, (-1, 3))
                int_floor_verts_in_tunnel_idxs = points_inside_of_tunnel_section(aps, tps, int_floor_verts)
                int_floor_verts_in_tunnel_idxs = np.reshape(np.array(int_floor_verts_in_tunnel_idxs), -1)
                for int_floor_vert_in_tunnel_idx in int_floor_verts_in_tunnel_idxs:
                    int_floor_vert_in_tunnel = int_floor_verts[int_floor_vert_in_tunnel_idx, :]
                    closest_ap_idx = np.argmin(np.linalg.norm(int_floor_vert_in_tunnel - aps, axis=1))
                    closest_ap_z = aps[closest_ap_idx, 2]
                    int_floor_vert_in_tunnel[2] = closest_ap_z + fta_dist
                    int_floor_verts[int_floor_vert_in_tunnel_idx, :] = int_floor_vert_in_tunnel
                    pb.update(1)
            floor_vertices[intersection_floor_vertices_idxs, :] = int_floor_verts
        vertices[floor_vertices_idxs, :] = floor_vertices
        log.info("Computing adjancency list")
        self.mesh.compute_adjacency_list()
        adjacency_list = self.mesh.adjacency_list
        log.info("Computing floor adj list")
        floor_adj_list = dict()
        pb = tqdm(total=len(floor_vertices_idxs), desc="Floor adj list")
        for vert_n, floor_idx in enumerate(floor_vertices_idxs):
            pb.update(1)
            floor_adj_list[floor_idx] = list()
            for adj_idx in adjacency_list[floor_idx]:
                if adj_idx in floor_vertices_idxs:
                    floor_adj_list[floor_idx].append(adj_idx)
        pb = tqdm(
            total=self._meshing_params.floor_smoothing_iter * len(floor_vertices_idxs),
            desc="Floor smoothing",
        )
        for n_iter in range(self._meshing_params.floor_smoothing_iter):
            copied_verts = np.copy(vertices)
            for vert_n, vert_idx in enumerate(floor_vertices_idxs):
                adj_verts_idxs = floor_adj_list[vert_idx]
                neigh_verts = copied_verts[adj_verts_idxs, :]
                if len(neigh_verts) > 0:
                    avg_z = np.average(neigh_verts[:, 2])
                    vertices[vert_idx, 2] = avg_z
                pb.update(1)
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)

    def save_mesh(self, path):
        o3d.io.write_triangle_mesh(path, self.mesh)

    @property
    def pyvista_mesh(self):
        o3d.io.write_triangle_mesh("temp.obj", self.mesh)
        pv_mesh = pv.read("temp.obj")
        os.remove("temp.obj")
        return pv_mesh

    def perlin_generator_of_tunnel(self, tunnel: Tunnel) -> CylindricalPerlinNoiseGenerator:
        return self._perlin_generator_of_tunnel[tunnel]


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(
    tunnel: Tunnel,
    perlin_generator: CylindricalPerlinNoiseGenerator,
    dist_between_circles,
    n_points_per_circle,
    radius,
    noise_magnitude,
    d_min=None,
    d_max=None,
):
    ads, aps, avs = tunnel.spline.discretize(dist_between_circles)
    n_circles = aps.shape[0]
    angs = np.linspace(0, 2 * np.pi, n_points_per_circle)
    angss = np.concatenate([angs for _ in range(n_circles)])
    angss = np.reshape(angss, [-1, 1])
    apss = np.concatenate(
        [np.full((n_points_per_circle, 3), aps[i, :]) for i in range(n_circles)],
        axis=0,
    )
    avss = np.concatenate(
        [np.full((n_points_per_circle, 3), avs[i, :]) for i in range(n_circles)],
        axis=0,
    )
    us = np.zeros(avs.shape)
    vs = np.zeros(avs.shape)
    for i, av in enumerate(avs):
        u, v = get_two_perpendicular_vectors(av)
        us[i, :] = np.reshape(u.cartesian_unitary, -1)
        vs[i, :] = np.reshape(v.cartesian_unitary, -1)
    uss = np.concatenate(
        [np.full((n_points_per_circle, 3), us[i, :]) for i in range(n_circles)],
        axis=0,
    )
    vss = np.concatenate(
        [np.full((n_points_per_circle, 3), vs[i, :]) for i in range(n_circles)],
        axis=0,
    )
    normals = uss * np.sin(angss) + vss * np.cos(angss)
    dss = np.concatenate(
        [np.full((n_points_per_circle, 1), ads[i, :]) for i in range(n_circles)],
        axis=0,
    )
    cylindrical_coords = np.concatenate([dss, angss], axis=1)
    noise_to_add = perlin_generator(cylindrical_coords)
    noise_to_add = np.reshape(noise_to_add, [-1, 1])
    noise_to_add /= np.max(np.abs(noise_to_add))
    noise_to_add -= 1
    noise_to_add /= 2
    points = apss + normals * radius * (1 + noise_magnitude * noise_to_add)
    return points, normals, aps, avs, apss, avss


def generate_noisy_sphere(
    center_point,
    radius,
    points_per_sm,
    perlin_mapper: SphericalPerlinNoiseMapper,
    noise_multiplier,
):
    area_of_sphere = 4 * np.pi * radius**2
    n_points = int(math.ceil(points_per_sm * area_of_sphere))
    points_before_noise = get_uniform_points_in_sphere(n_points)
    noise_of_points = np.reshape(perlin_mapper(points_before_noise), (-1, 1))
    points_with_noise = points_before_noise + noise_multiplier * noise_of_points * points_before_noise
    points_with_noise_and_radius = points_with_noise * radius
    normals = points_with_noise_and_radius / np.reshape(np.linalg.norm(points_with_noise_and_radius, axis=1), (-1, 1))
    points = points_with_noise_and_radius + center_point
    return points, normals


def points_inside_of_tunnel_section(
    axis_points: np.ndarray,
    tunnel_points: np.ndarray,
    points: np.ndarray,
    axis_vectors: np.ndarray = None,
    max_projection_over_axis=0.5,
):
    try:
        assert axis_points.shape[0] > 0
        assert axis_points.shape[1] == 3
        assert tunnel_points.shape[0] > 0
        assert tunnel_points.shape[1] == 3
        assert points.shape[0] > 0
        assert points.shape[1] == 3
    except:
        return np.array([], dtype=int)
    dist_of_ps_to_aps = distance_matrix(points, axis_points)
    closest_ap_to_p_idx = np.argmin(dist_of_ps_to_aps, axis=1)
    if not axis_vectors is None:
        closest_ap_to_p_vector = points - axis_points[closest_ap_to_p_idx, :]
        ap_vector_of_p = axis_vectors[closest_ap_to_p_idx, :]
        projection = np.array([np.dot(ap_vector_of_p[i, :], closest_ap_to_p_vector[i, :]) for i in range(ap_vector_of_p.shape[0])])
        outside_because_of_angle = np.abs(projection) > max_projection_over_axis
    else:
        outside_because_of_angle = np.zeros((points.shape[0]), dtype=np.bool8)
    dist_of_ps_to_tps = distance_matrix(points, tunnel_points)
    dist_of_tps_to_aps = distance_matrix(tunnel_points, axis_points)
    dist_to_ap_of_p = np.min(dist_of_ps_to_aps, axis=1)
    closest_tp_to_p = np.argmin(dist_of_ps_to_tps, axis=1)
    inside_for_radius = dist_to_ap_of_p < dist_of_tps_to_aps[closest_tp_to_p, closest_ap_to_p_idx]
    inside = np.logical_and(inside_for_radius, np.logical_not(outside_because_of_angle))
    return np.where(inside)


def ids_points_inside_ptcl_sphere(sphere_points, center_point, points):
    dists_of_ps_to_sph_pts = distance_matrix(points, sphere_points)
    dists_of_sph_pts_to_center = np.reshape(np.linalg.norm(sphere_points - center_point, axis=1), (-1, 1))
    dists_of_ps_to_center = np.reshape(np.linalg.norm(points - center_point, axis=1), (-1, 1))
    id_closest_sph_pt_to_pt = np.argmin(dists_of_ps_to_sph_pts, axis=1)
    return np.where(dists_of_ps_to_center < dists_of_sph_pts_to_center[id_closest_sph_pt_to_pt])
