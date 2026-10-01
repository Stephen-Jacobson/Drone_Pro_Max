import numpy as np
import matplotlib.pyplot as plt

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.tunnel import *
import random


def main():
    # Generate the graph
    fig = plt.figure(figsize=(10, 10))
    graph = TunnelNetwork()
    central_node = CaveNode()
    for i in range(10):
        tunnel_params = TunnelParams(
            {
                "distance": np.random.uniform(100, 100),
                "starting_direction": angles_to_vector(
                    (
                        np.random.uniform(0, np.pi * 2),
                        np.random.uniform(np.deg2rad(-10), np.deg2rad(10)),
                    )
                ),
                "horizontal_tendency": np.deg2rad(np.random.uniform(-30, 30)),
                "horizontal_noise": np.deg2rad(5),
                "vertical_tendency": np.deg2rad(np.random.uniform(-10, 10)),
                "vertical_noise": np.deg2rad(3),
                "segment_length": 30,
                "segment_length_noise": 5,
                "node_position_noise": 4,
            }
        )
        if i == 0:
            success = Tunnel(
                graph, initial_node=central_node, params=tunnel_params
            ).success
            continue
        success = False
        while not success:
            node = random.choice(graph.nodes)
            if len(node.connected_nodes) < 2:
                print(node.connected_nodes)
                continue
            success = Tunnel(graph, initial_node=node, params=tunnel_params).success
    for i in range(4):
        success = False
        while not success:
            i_node = random.choice(graph.nodes)
            f_node = random.choice(graph.nodes)
            assert isinstance(i_node, CaveNode)
            assert isinstance(f_node, CaveNode)
            if i_node is f_node:
                continue
            same_tunnel = False
            for tunnel in i_node.tunnels:
                if tunnel in f_node.tunnels:
                    same_tunnel = True
                    break
            if same_tunnel:
                continue
            # debug_plot(graph, wait=False)
            # plt.scatter(i_node.x, i_node.y, c="r", s=1000)
            # plt.scatter(f_node.x, f_node.y, c="k", s=1000)
            # plt.draw()
            # plt.waitforbuttonpress()
            last_tunnel = Tunnel(
                graph,
                initial_node=i_node,
                final_node=f_node,
                params=tunnel_params,
            )
            success = last_tunnel.success
    # debug_plot(graph)
    # jwith open("datafiles/graph.pkl", "wb") as f:
    #    pickle.dump(graph, f)


if __name__ == "__main__":
    main()
