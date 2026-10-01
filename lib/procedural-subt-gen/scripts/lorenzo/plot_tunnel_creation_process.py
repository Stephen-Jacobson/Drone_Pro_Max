import os
import pathlib
import pickle
import pyvista as pv
import numpy as np
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, CaveNode
from subt_proc_gen.mesh_generation import (
    TunnelNetworkWithMesh,
    TunnelPtClGenParams,
    TunnelWithMesh,
)
from subt_proc_gen.geometry import Spline3D
import imageio
import pygifsicle
import cv2
import logging

colors = [
    "#0000FF",
    "#1589FF",
    "#008000",
    "#004225",
    "#004225",
    "#E2F516",
    "#EE9A4D",
    "#FD1C03",
    "#800517",
    "#E238EC",
    "#800080",
]
logging.basicConfig(level=logging.DEBUG)
base_video_folder = os.path.join(pathlib.Path.home(), "Documents/my_papers/IROS2023/video/tunnel_creation_process")
folder_g = "/home/lorenzo/git/procedural-subt-gen/datafiles"
pth_g = os.path.join(folder_g, "graph_for_plotting.pkl")
pth_gnn = os.path.join(folder_g, "graph_for_plotting_no_noise.pkl")
pth_gn = os.path.join(folder_g, "graph_for_plotting_noise.pkl")


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


class PlottingParameters:
    def __init__(self):
        self.radius_of_nodes = 0.7
        self.radius_of_edges = 0.3
        self.radius_of_splines = 0.2
        self.radius_of_points = 0.09


P = PlottingParameters()


class MyPlotter:
    def __init__(self):
        self.inactive_render = False
        self.n_captures_per_change = 3
        self.azimuth_increment = 0.4
        self.elevation_increment = 0
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background(color="white")
        self.n_image = 0
        self.n_trial = len(os.listdir(base_video_folder))
        self.images_folder = os.path.join(base_video_folder, f"trial_n_{self.n_trial}")
        print(self.images_folder)
        os.mkdir(self.images_folder)
        self.set_camera_params()
        self.images = None

    def plot_node(self, node: CaveNode, color, capture_at_end=True):
        sphere = pv.Sphere(radius=P.radius_of_nodes, center=node.xyz)
        actor = self.plotter.add_mesh(sphere, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def plot_edge(self, n1: CaveNode, n2: CaveNode, color, capture_at_end=True):
        tube = pv.Tube(pointa=n1.xyz, pointb=n2.xyz, radius=P.radius_of_edges)
        actor = self.plotter.add_mesh(tube, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def plot_spline(self, spline: Spline3D, color, capture_at_end=True):
        msh_of_spl = polyline_from_points(spline.discretize(0.5)[1]).tube(radius=P.radius_of_splines)
        actor = self.plotter.add_mesh(msh_of_spl, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def plot_o3d_mesh(self, mesh, color, capture_at_end=True):
        v = np.asarray(mesh.vertices)
        f = np.array(mesh.triangles)
        f = np.c_[np.full(len(f), 3), f]
        mesh = pv.PolyData(v, f)
        actor = self.plotter.add_mesh(mesh, show_edges=True, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def plot_tunnel_nodes(self, tunnel: Tunnel, color, capture_at_end=True):
        n_nodes = len(tunnel.nodes)
        node_actors = []
        for n_node, node in enumerate(tunnel.nodes):
            node_actors.append(self.plot_node(node, color=color, capture_at_end=capture_at_end))
        return node_actors

    def plot_tunnel_nodes_and_edges(self, tunnel: Tunnel, capture_at_end=True):
        n_nodes = len(tunnel.nodes)
        color = "r" if tunnel.tunnel_type == "grown" else "b"
        node_actors = []
        edge_actors = []
        for n_node, node in enumerate(tunnel.nodes):
            node_actors.append(self.plot_node(node, color=color, capture_at_end=capture_at_end))
            if n_node < n_nodes - 1:
                edge_actors.append(
                    self.plot_edge(
                        node,
                        tunnel.nodes[n_node + 1],
                        color=color,
                        capture_at_end=capture_at_end,
                    )
                )
        return node_actors, edge_actors

    def plot_tunnel_spline(self, tunnel: Tunnel, color, capture_at_end=True):
        return self.plot_spline(tunnel.spline, color, capture_at_end=capture_at_end)

    def plot_points_of_tunnel(self, tunnel: TunnelWithMesh, color, capture_at_end=True):
        points = tunnel.all_selected_points
        point_actors = []
        mesh = pv.PolyData(points)
        glyph_points = mesh.glyph(orient=False, scale=False, geom=pv.Sphere(radius=P.radius_of_points))
        point_actors.append(self.plotter.add_mesh(glyph_points, color=color))
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return point_actors

    def save_capture(self):
        if self.inactive_render:
            return
        # path_to_image = os.path.join(self.images_folder, f"{self.n_image:04d}.png")
        path_to_image = os.path.join(self.images_folder, f"last_image.png")
        self.plotter.show(auto_close=False, screenshot=path_to_image)
        self.plotter.camera.azimuth += self.azimuth_increment
        self.plotter.camera.elevation += self.elevation_increment
        self.n_image += 1
        if self.images is None:
            self.images = [
                cv2.imread(path_to_image),
            ]
        else:
            self.images.append(cv2.imread(path_to_image))

    def print_camera_params(self):
        print(f"self.plotter.camera.position = {self.plotter.camera.position}")
        print(f"self.plotter.camera.focal_point= {self.plotter.camera.focal_point}")
        print(f"self.plotter.camera.roll = {self.plotter.camera.roll}")

    def set_camera_params(self):
        self.plotter.camera.position = (
            104.5786207334545,
            111.92568096131357,
            86.94361170474676,
        )
        self.plotter.camera.focal_point = (
            -8.070702349350167,
            -0.8108855992463173,
            13.159967378983225,
        )
        self.plotter.camera.roll = -113.67940228246788

    def remove_actors(self, actors):
        assert isinstance(actors, list)
        for actor in actors:
            self.plotter.remove_actor(actor)

    def create_gif(self):
        logging.info("Creating gif")
        path_to_gif = os.path.join(self.images_folder, "movie.gif")
        imageio.mimsave(path_to_gif, self.images, fps=24)
        pygifsicle.optimize(path_to_gif)

    def create_video(self):
        logging.info("Creating Video")
        path_to_video_file = os.path.join(self.images_folder, "video.avi")
        height, width, layers = self.images[0].shape
        self.video_writer = cv2.VideoWriter(path_to_video_file, 0, 24, (width, height))
        for frame in self.images:
            self.video_writer.write(frame)
        cv2.destroyAllWindows()
        self.video_writer.release()

    def clear_all(self):
        self.plotter.clear()


def main():
    my_plotter = MyPlotter()
    print("Loading graph")
    with open(pth_g, "rb") as f:
        tnn = pickle.load(f)
        assert isinstance(tnn, TunnelNetwork)
    print("Loading mesh graph with no noise")
    with open(pth_gnn, "rb") as f:
        tnn_nn = pickle.load(f)
        assert isinstance(tnn_nn, TunnelNetworkWithMesh)
    print("Loading mesh graph with noise")
    with open(pth_gn, "rb") as f:
        tnn_n = pickle.load(f)
        assert isinstance(tnn_n, TunnelNetworkWithMesh)
    print("Plotting the network graph creation")
    node_actors, edge_actors = [], []
    for n_tunnel, tunnel in enumerate(tnn.tunnels):
        print(f"{n_tunnel+1:04d}", end="\r", flush=True)
        assert isinstance(tunnel, Tunnel)
        iter_node_actors, iter_edge_actors = my_plotter.plot_tunnel_nodes_and_edges(tunnel)
        node_actors.append(iter_node_actors)
        edge_actors.append(iter_edge_actors)
    print("")
    # Delete everything, replot with other color and the splines
    my_plotter.azimuth_increment = 0
    my_plotter.elevation_increment = 0.4
    P.radius_of_nodes = 0.5
    print("Splines")
    my_plotter.n_captures_per_change = 12
    for n_tunnel, tunnel in enumerate(tnn.tunnels):
        print(f"{n_tunnel+1:04d}", end="\r", flush=True)
        color = colors[n_tunnel]
        my_plotter.remove_actors(node_actors[n_tunnel])
        my_plotter.remove_actors(edge_actors[n_tunnel])
        my_plotter.plot_tunnel_nodes(tunnel, color, capture_at_end=False)
        my_plotter.plot_tunnel_spline(tunnel, color)
    print("")
    _ = [my_plotter.save_capture() for _ in range(my_plotter.n_captures_per_change)]
    my_plotter.elevation_increment = 0
    print("Tunnels with no noise")
    points_with_no_noise_actors = []
    for n_tunnel, tunnel_with_mesh in enumerate(tnn_nn._tunnels_with_mesh):
        print(f"{n_tunnel+1:04d}", end="\r", flush=True)
        color = colors[n_tunnel]
        points_with_no_noise_actors.append(my_plotter.plot_points_of_tunnel(tunnel_with_mesh, color=color))
    print("")
    # Plot all the the the points in the different tunnels
    print("Tunnels with noise")
    points_with_noise_actors = []
    for n_tunnel, tunnel_with_mesh in enumerate(tnn_n._tunnels_with_mesh):
        print(f"{n_tunnel+1:04d}", end="\r", flush=True)
        color = colors[n_tunnel]
        my_plotter.remove_actors(points_with_no_noise_actors[n_tunnel])
        points_with_noise_actors.append(my_plotter.plot_points_of_tunnel(tunnel_with_mesh, color=color))
    print("")
    print("Plotting the intersection cleaning")
    for n_intersection, intersection in enumerate(tnn_n._tunnel_network.intersections):
        print(f"{n_intersection+1:04d}", end="\r", flush=True)
        tnn_n.clean_intersection(intersection)
        for tunnel in intersection.tunnels:
            twm = tnn_n.tunnel_to_tunnel_with_mesh[tunnel]
            idx = tnn_n._tunnels_with_mesh.index(twm)
            color = colors[idx]
            my_plotter.remove_actors(points_with_noise_actors[idx])
            points_with_noise_actors[idx] = my_plotter.plot_points_of_tunnel(twm, color)
    my_plotter.elevation_increment = -1
    for _ in range(40):
        my_plotter.save_capture()
    my_plotter.elevation_increment = 0
    my_plotter.azimuth_increment = 0
    print("Creating the mesh")
    mesh, simplified_mesh = tnn_n.compute_mesh()
    print("Plotting the mesh")
    my_plotter.clear_all()
    my_plotter.plot_o3d_mesh(simplified_mesh, color="white")
    my_plotter.azimuth_increment = 2
    for _ in range(180):
        my_plotter.save_capture()
    print("Creating the video")
    my_plotter.create_video()


if __name__ == "__main__":
    main()
