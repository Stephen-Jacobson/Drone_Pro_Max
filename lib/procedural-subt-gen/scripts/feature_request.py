from subt_proc_gen.tunnel import Tunnel, TunnelNetwork, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
import numpy as np
import random
from networkx import Graph, has_path, is_connected
from subt_proc_gen.param_classes import TunnelPtClGenParams
from pyvista.plotting.plotting import Plotter
import pyvista as pv
from subt_proc_gen.display_functions import plot_graph, plot_node, plot_mesh
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.DEBUG)


def gen_raw_nodes(
    height: int = 5, width: int = 5, distance: float = 50, z_variation: tuple[float] = None
):
    # Generates a set of nodes in a matrix-like manner
    if z_variation is None:
        z_variation = (0.0, 0.0)
    zv = z_variation
    nodes = {}
    for h in range(height):
        for w in range(width):
            nodes[(h, w)] = Node(
                h * distance, w * distance, np.random.random() * (zv[1] - zv[0]) + zv[0]
            )
    return nodes


# -----------------------------------------------------------------------------------------------------------------------------------
def create_tunnels_from_nodes(nodes: dict):
    tunnels = []
    tunnel_set = set()
    for h, w in nodes.keys():
        node = nodes[(h, w)]
        for hv, wv in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if hv == 0 and wv == 0:
                continue
            h_ = int(h + hv)
            w_ = int(w + wv)
            try:
                node_ = nodes[(h_, w_)]
            except:
                continue
            tunnel_identifier = {(h, w), (h_, w_)}
            if tuple(tunnel_identifier) in tunnel_set:
                continue
            tunnel_set.add(tuple(tunnel_identifier))
            tunnels.append(Tunnel.connector(inode=node, fnode=node_))
    return tunnels


def tn_to_networkx(tn: TunnelNetwork):
    nodes = list(tn.nodes)
    edges = list(tn.edges)
    node_to_node_id = dict()
    for node_id, node in enumerate(nodes):
        node_to_node_id[node] = node_id
    graph = Graph()
    for node in nodes:
        graph.add_node(node_to_node_id[node])
    for edge in edges:
        n1, n2 = edge.nodes
        nid1 = node_to_node_id[n1]
        nid2 = node_to_node_id[n2]
        graph.add_edge(nid1, nid2)
    return graph, node_to_node_id


def remove_tunnels_mantaining_connectivity(tn, n_to_remove, node1, node2):
    n_removed = 0
    tunnels_removed = []
    while n_removed < n_to_remove:
        removed_tunnel = random.choice(list(tn.tunnels))
        if node1 in removed_tunnel.nodes or node2 in removed_tunnel.nodes:
            continue
        tn.remove_tunnel(removed_tunnel)
        graph, node_to_id = tn_to_networkx(tn)
        try:
            nodei_id = node_to_id[node1]
            nodef_id = node_to_id[node2]
            if has_path(graph, nodei_id, nodef_id) and is_connected(graph):
                n_removed += 1
                tunnels_removed.append(removed_tunnel)
            else:
                tn.add_tunnel(removed_tunnel)
        except:
            tn.add_tunnel(removed_tunnel)
    return tunnels_removed


def save_img_of_env(
    path, nodei: Node, nodef: Node, tn: TunnelNetwork, tnmg: TunnelNetworkMeshGenerator
):
    plotter = Plotter(off_screen=True)
    plotter.set_background("w")
    plot_node(plotter, nodei, radius=8, color="r")
    plot_node(plotter, nodef, radius=8, color="r")
    # plot_graph(plotter, tn)
    plot_mesh(plotter, tnmg, style="wireframe")
    plotter.show(screenshot=path)


class Camera:
    def __init__(self):
        self.data = None


camara = Camera()


def main(n):
    for i in range(n):
        if i == 0:
            nodes = gen_raw_nodes(5, 5, 50, (0, 0))
            nodei = nodes[(0, 1)]
            nodef = nodes[(4, 4)]
            tunnels = create_tunnels_from_nodes(nodes)
            random.shuffle(tunnels)
            tn = TunnelNetwork(initial_node=False)
            for tunnel in tunnels:
                tn.add_tunnel(tunnel)
            TunnelPtClGenParams._random_radius_interval = (3, 7)
            tnmg = TunnelNetworkMeshGenerator(tn)
            tnmg.compute_all(compute_floors=False)
            save_img_of_env(
                f"/home/lorenzo/images/papers/subt_proc_gen/base.png", nodei, nodef, tn, tnmg
            )
        tunnels_removed = remove_tunnels_mantaining_connectivity(tn, 15, nodei, nodef)
        tnmg._tunnel_network = tn
        tnmg.compute_mesh()
        save_img_of_env(
            f"/home/lorenzo/images/papers/subt_proc_gen/{i:010d}.png", nodei, nodef, tn, tnmg
        )
        for tunnel in tunnels_removed:
            tn.add_tunnel(tunnel)


if __name__ == "__main__":
    main(10)
