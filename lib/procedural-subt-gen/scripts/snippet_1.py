"""
This script shows the three main different ways of creating a Tunnel network:
    - Creating the nodes manually.
    - Generating a Grown tunnel.
    - Generating a Connector tunnel.
"""
# Imports
from subt_proc_gen.tunnel import (
    Node,
    Tunnel,
    TunnelNetwork,
    TunnelNetworkParams,
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.display_functions import plot_graph, plot_splines, plot_xyz_axis
from subt_proc_gen.geometry import Vector3D
import pyvista as pv
import numpy as np
import cv2


def main():
    """
    We first create an instance of TunnelNetwork. This class is a wraper
    around the 'Graph' class, that allows us to add complexity to the graph
    by adding tunnels.

    It also ensures that no collisions between tunnels happen when adding random tunnels,
    and that the random tunnels adhere to some other constraints. These are constraints are
    controlled with the 'TunnelNetworkParams' class.
    """
    # Definition of the params is optional, defaults are available by not feeding any params into the 'TunnelNetwork' class at creation.
    tunnel_network_params = TunnelNetworkParams(
        collision_distance=10,  # (Meters) (Only in procedural tunnels) Determines how close two Tunnel splines can be
        max_inclination_rad=np.deg2rad(20),  # (Radians) (Only In prodedural tunnels) Determines the maximum inclination a tunnel can have at any point in its spline
        min_intersection_angle_rad=np.deg2rad(40),  # (Radians) (Only in procedural tunnels) Determines how parallel can two tunnels be at their intersection (By consecuence, controlls how many tunnels can intersect at the same node)
        min_distance_between_intersections=20,  # (Meters) (Only in procedural tunnels) Controlls how close two intersection nodes can be
    )
    tn = TunnelNetwork(
        tunnel_network_params,
        initial_node=False,  # Create the TunnelNetwork without a node already in (0,0,0)
    )
    """The first way of creating a tunnel is to set the nodes manually"""
    node1 = Node(0, 0, 0)
    node2 = Node(60, 0, 0)
    node3 = Node(0, 60, 0)
    tunnel1 = Tunnel((node1, node2, node3))
    tn.add_tunnel(tunnel1)
    """The second way of creating a tunnel is to use a 'connector' tunnel"""
    connector_tunnel_params = ConnectorTunnelGenerationParams(
        segment_length=10,  # The maximum distance between the nodes of the tunnel
        node_position_horizontal_noise=5,  # Maximum horizontal displacement of the nodes from the segment that goes from inode to fnode
        node_position_vertical_noise=3,  # Maximum vertical displacement of the nodes from the segment that goes from inode to fnode
    )
    tunnel2 = Tunnel.connector(
        inode=node1,
        fnode=node3,
        params=connector_tunnel_params,  # This is optional, defaults available
    )
    tn.add_tunnel(tunnel2)
    if tn.check_collisions(tunnel2):  # Checks if tunnel2 collides with any other tunnel in the TunnelNetwork
        tn.remove_tunnel(tunnel2)
    """The third way of creating a tunnel is to 'grow' one"""
    grown_tunnel_params = GrownTunnelGenerationParams(
        distance=100,  # The length of the generated tunnel
        horizontal_tendency_rad=np.deg2rad(-20),  # The average horizontal angle between sequential segments of the tunnel
        vertical_tendency_rad=np.deg2rad(-10),  # The average vertical angle of the segments of the tunnel
        horizontal_noise_rad=np.deg2rad(5),  # The deviation of the horizontal angle between the sequential segments of the tunnel
        vertical_noise_rad=np.deg2rad(5),  # The deviation of the vertical angle of the segments of the tunnel
        min_segment_length=10,  # The minimum distance between two sequential nodes of the tunnel
        max_segment_length=15,  # The maximum distance between two sequential nodes of the tunnel
    )
    tunnel3 = Tunnel.grown(
        i_node=node2,  # The node from which the tunnel is going to be grown
        i_direction=Vector3D((1, 0, 0)),  # It is a good idea to set the initial direction, if it is random, the probabilities of collision are high
        params=grown_tunnel_params,  # Optional, defaults available
    )
    tn.add_tunnel(tunnel3)
    if tn.check_collisions(tunnel3):  # Checks if tunnel2 collides with any other tunnel in the TunnelNetwork
        tn.remove_tunnel(tunnel3)
    #######################################################################################################################
    # Show the results
    #######################################################################################################################
    plotter = pv.Plotter(window_size=(1920, 1080))
    plot_graph(plotter, tn)
    plot_splines(plotter, tn, radius=1)
    plot_xyz_axis(plotter)
    plotter.add_text(
        text="""White: Spline of tunnel from nodes
Red: Spline of Connector tunnel
Blue: Spline of Grown tunnel
Black dots: Nodes
Black lines: Edges""",
        font_size=10,
    )
    cpos, image = plotter.show(return_img=True, return_cpos=True, screenshot=True)
    # cv2.imwrite("README/snippet1.png", image)


if __name__ == "__main__":
    main()
