"""
This script details how to generate the pointcloud and the mesh from a TunnelNetwork
"""
# Imports
from subt_proc_gen.tunnel import (
    Node,
    Tunnel,
    TunnelNetwork,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkPtClGenStrategies,
    TunnelNetworkMeshGenParams,
    TunnelPtClGenParams,
)
from subt_proc_gen.perlin import PerlinParams
from subt_proc_gen.display_functions import plot_graph, plot_splines, plot_xyz_axis, plot_intersection_ptcls, plot_tunnel_ptcls, plot_mesh
import pyvista as pv
import cv2


def main():
    #######################################################################################################################
    # CREATE THE TUNNEL NETWORK
    #######################################################################################################################
    node1 = Node(0, 0, 0)
    node2 = Node(60, 0, 0)
    node3 = Node(60, 60, 0)
    node4 = Node(0, 60, 0)
    tunnel1 = Tunnel((node1, node2))
    tunnel2 = Tunnel((node2, node3))
    tunnel3 = Tunnel((node3, node4))
    tunnel4 = Tunnel((node1, node3))
    tn = TunnelNetwork(initial_node=False)
    tn.add_tunnel(tunnel1)
    tn.add_tunnel(tunnel2)
    tn.add_tunnel(tunnel3)
    tn.add_tunnel(tunnel4)
    #######################################################################################################################
    # SET THE PARAMETERS
    #######################################################################################################################
    """ For the pointcloud generation process it is important to note that all parameters must be set before starting the generation process"""
    pre_set_tunnel_params = dict()
    pre_set_tunnel_params[tunnel1] = TunnelPtClGenParams(  # setting the parameters of tunnel1
        dist_between_circles=None,  # Using default value
        n_points_per_circle=None,  # Using default value
        radius=5,  # Radius of the cross section of the tunnel
        noise_multiplier=0.5,  # Magnitude of the noise
        perlin_params=PerlinParams(res=3, octaves=3, persitence=2, lacunarity=2),  # these parameters control the properties of the perlin noise applied, for more info see
    )
    pre_set_tunnel_params[tunnel2] = TunnelPtClGenParams(  # setting the parameters of tunnel2
        dist_between_circles=None,
        n_points_per_circle=None,
        radius=2,
        noise_multiplier=0.5,
        perlin_params=PerlinParams.random(),  # Random perlin noise
    )
    pre_set_tunnel_params[tunnel3] = TunnelPtClGenParams.random()  # This randomly set all paramters (within reasonable limits)
    """This class is the one that controlls the pointcloud generation process"""
    ptcl_generation_params = TunnelNetworkPtClGenParams(
        ptcl_gen_strategy=TunnelNetworkPtClGenStrategies.default,  # The tunnels that dont have pre-set parameters, this decides if a random set of parameters or the default parameters will be used (in this case, the tunnel4 will use the default paramters)
        # ptcl_gen_strategy=TunnelNetworkPtClGenStrategies.random, # This would make the tunnel4 have random parameters, the rest of the tunnels are in the pre_set_tunnel_params dictionary
        pre_set_tunnel_params=pre_set_tunnel_params,  # This is used to set the ptcl gen params of an specific tunnel(s)
    )
    """This class controlls how the mesh will be generated from the """
    meshing_params = TunnelNetworkMeshGenParams(
        poisson_depth=11,  # Argument of the meshing algorithm, 11 works fine
        simplification_voxel_size=None,  # (using default) If the mesh is simplified with the vertex clustering method
        voxelization_voxel_size=None,  # (using default) Relevant to the floor smoothing part.
        fta_distance=-1,  # Floor to axis distance
        floor_smoothing_iter=5,  # How many iterations of smoothing the floor to use
        floor_smoothing_r=None,  # (Using default) Radius of smoothing
    )
    #######################################################################################################################
    # GENERATE THE PTCL AND MESH
    #######################################################################################################################
    ptcl_and_mesh_generator = TunnelNetworkMeshGenerator(tunnel_network=tn, ptcl_gen_params=ptcl_generation_params, meshing_params=meshing_params)
    """Next line is an alternative way of assigning random parameters that takes much less code"""
    # ptcl_and_mesh_generator = TunnelNetworkMeshGenerator(tunnel_network=tn, ptcl_gen_params=TunnelNetworkPtClGenParams.random(), meshing_params=TunnelNetworkMeshGenParams.random())
    ptcl_and_mesh_generator.compute_all()
    #######################################################################################################################
    # Show the results
    #######################################################################################################################
    plotter = pv.Plotter(window_size=(1920, 1080))
    plot_xyz_axis(plotter)
    plot_splines(plotter, tn, radius=1)
    plot_tunnel_ptcls(plotter, ptcl_and_mesh_generator, color="r")
    plot_intersection_ptcls(plotter, ptcl_and_mesh_generator, color="b")
    plot_mesh(plotter, ptcl_and_mesh_generator)
    plotter.add_text(
        text="""Red Dots: Intersection Points
Blue Dots: Tunnel Points""",
        font_size=10,
    )
    cpos, image = plotter.show(return_img=True, return_cpos=True, screenshot=True)
    # cv2.imwrite("README/snippet3_1.png", image)


if __name__ == "__main__":
    main()
