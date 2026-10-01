import pyvista as pv
import numpy as np
import os

for i in [5]:
    for j in range(4):
        path_to_mesh = f"datafiles/same_network_different_mesh_{i}_{j}_simplified.obj"
        path_to_screenshot = os.path.splitext(path_to_mesh)[0] + ".png"
        path_to_screenshot_wireframe = (
            os.path.splitext(path_to_mesh)[0] + "_wireframe" + ".png"
        )
        path_to_screenshot_wireframe_1 = (
            os.path.splitext(path_to_mesh)[0] + "_wireframe1" + ".png"
        )

        mesh = pv.get_reader(path_to_mesh).read()
        if j == 0:
            plotter = pv.Plotter()
            plotter.set_background("white")
            plotter.add_mesh(mesh, edge_color="black", color="gray", show_edges=True)
            plotter.show()
            cpos = plotter.camera.position
            cfpt = plotter.camera.focal_point
            cr = plotter.camera.roll
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            edge_color="black",
            color="gray",
            show_edges=True,
        )
        plotter.camera.position = cpos
        plotter.camera.focal_point = cfpt
        plotter.camera.roll = cr
        plotter.show(screenshot=path_to_screenshot)
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.disable_anti_aliasing()
        plotter.add_mesh(mesh, show_edges=True)
        plotter.camera.position = cpos
        plotter.camera.focal_point = cfpt
        plotter.camera.roll = cr
        plotter.show(screenshot=path_to_screenshot_wireframe)
        plotter = pv.Plotter(off_screen=True, lighting="none")
        plotter.set_background("white")
        plotter.disable_anti_aliasing()
        plotter.add_mesh(mesh, show_edges=True)
        plotter.camera.position = cpos
        plotter.camera.focal_point = cfpt
        plotter.camera.roll = cr
        plotter.show(screenshot=path_to_screenshot_wireframe_1)
