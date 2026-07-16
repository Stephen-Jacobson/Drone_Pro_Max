"""
This script showcases how to use the tunnel generators inside the TunnelNetwork class to procedurally create arbitrarily large tunnel networks.
"""
# Imports
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.display_functions import plot_graph, plot_splines, plot_xyz_axis
import pyvista as pv
import cv2


def main():
    # PARAMETERS
    number_of_grown_tunnels = 10
    number_of_connector_tunnels = 5
    """
    We first create an instance of TunnelNetwork.
    """
    tn = TunnelNetwork()
    """
    Then we generate the grown tunnels
    """
    for i in range(number_of_grown_tunnels):
        print(f"Generating grown tunnel {i+1} out of {number_of_grown_tunnels}")
        params = GrownTunnelGenerationParams.random()
        tn.add_random_grown_tunnel(  # This function handles the selection of the initial node and initial direction, as well as checking the collisions
            params,
            n_trials=1000,  # If the checks of the generated tunnel fail, it is possible to try again as many times as necessary to generate a tunnel that does not collide
        )
    for j in range(number_of_connector_tunnels):
        print(f"Generating connector tunnel {j+1}, out of {number_of_connector_tunnels}")
        params = ConnectorTunnelGenerationParams.random()
        tn.add_random_connector_tunnel(  # This function handles the selection of the initial and final node, as well as the checks
            params,
            n_trials=1000,  # If the checks of the generated tunnel fail, it is possible to try to generate another one as many times as necessary, until it passes all checks
        )
    #######################################################################################################################
    # Show the results
    #######################################################################################################################
    plotter = pv.Plotter(window_size=(1920, 1080))
    plot_graph(plotter, tn)
    plot_splines(plotter, tn, radius=1)
    plot_xyz_axis(plotter)
    plotter.add_text(
        text="""Red: Spline of Connector tunnel
Blue: Spline of Grown tunnel
Black dots: Nodes
Black lines: Edges""",
        font_size=10,
    )
    cpos, image = plotter.show(return_img=True, return_cpos=True, screenshot=True)
    # cv2.imwrite("README/snippet2_1.png", image)


if __name__ == "__main__":
    main()
